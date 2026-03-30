from collections import defaultdict
import json
import random
import time
import os
import pickle
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Dict, Any, Callable

from tqdm import tqdm
from transformers import AutoTokenizer
import dotenv

# --- CUSTOM IMPORTS ---
from config import K_SCHEDULE
from engines import HybridRAGEngine, TextRAGEngine, GraphRAGEngine, BaselineEngine
from judge import LLMJudge, DualJudge
from llms import HybridFactDeductor, RetrievalCache, TogetherFactDeductorV3
from utils.parse_utils import load_passages

# Load API keys
dotenv.load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DATA_FILE = "results/questions.jsonl"
OUTPUT_FILE = "results/experiment_results.jsonl"
CACHE_DIR = "engine_cache"

# Parallelization settings
# Together AI rate limits: ~60-600 RPM depending on tier
# Conservative default; increase if you have higher limits
MAX_WORKERS = 8

# Thread-safe locks
write_lock = threading.Lock()
finished_lock = threading.Lock()

# ------------------------------
# MODELS (All OSS except one)
# ------------------------------
MODELS = {
    "small": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "efficient": "openai/gpt-oss-20b",
    "medium": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "large": "openai/gpt-oss-120b",
    "proprietary": "gemini-2.5-pro",  # Accuracy-only baseline
    "small_proprietary": "gemini-3-flash-preview"  # Skipped
}

# ------------------------------
# COST MATRIX (per 1M tokens)
# Together-style pricing
# ------------------------------
COST_MATRIX = {
    # flat rate
    "small": {"input": 0.18, "output": 0.18},
    "medium": {"input": 0.88, "output": 0.88},
    # openai/gpt-oss-* models
    "efficient": {"input": 0.05, "output": 0.20},
    "large": {"input": 0.15, "output": 0.60},
    # proprietary baselined/. 
    "proprietary": {"input": 1.25, "output": 10.0},
    "small_proprietary": {"input": 0.5, "output": 3.0},
}

print("Loading Tokenizer for Metrics...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ==========================================
# TOKEN HELPERS
# ==========================================

def count_tokens(text: Any) -> int:
    if text is None:
        return 0
    # Coerce common non-string types to a string representation
    if isinstance(text, (list, dict)):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)
    elif not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


# ==========================================
# COST HELPER
# ==========================================


def compute_cost(model_category: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost based on real Together pricing."""
    pricing = COST_MATRIX[model_category]

    in_rate = float(pricing["input"])
    out_rate = float(pricing["output"])

    cost_in = (input_tokens / 1_000_000) * in_rate
    cost_out = (output_tokens / 1_000_000) * out_rate

    return cost_in + cost_out


# ==========================================
# ENGINE CACHING
# ==========================================


def get_cached_engine(engine_name: str, factory_func: Callable):
    os.makedirs(CACHE_DIR, exist_ok=True)
    file_path = os.path.join(CACHE_DIR, f"{engine_name}.pkl")

    if os.path.exists(file_path):
        print(f"⚡ Loading {engine_name} from cache...")
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Cache corrupted ({e}). Rebuilding...")

    print(f"🔨 Building {engine_name}...")
    engine_instance = factory_func()

    print(f"💾 Saving {engine_name}...")
    try:
        with open(file_path, "wb") as f:
            pickle.dump(engine_instance, f)
    except Exception as e:
        print(f"⚠️ Failed to cache engine ({e})")

    return engine_instance


# ==========================================
# LOAD DATA
# ==========================================


def load_evaluation_dataset(
    filepath: str, limit_per_category: int = None
) -> List[Dict[str, Any]]:

    if not os.path.exists(filepath):
        print(f"❌ File missing: {filepath}")
        return []

    category_buckets = defaultdict(list)

    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if "question" in item:
                    cat = item.get("category", "Unknown").title().strip()
                    if cat == "Multi-Hop":
                        cat = "Multi-hop"
                    category_buckets[cat].append(item)
            except json.JSONDecodeError:
                continue

    final_dataset = []
    random.seed(42)

    for cat, questions in category_buckets.items():
        random.shuffle(questions)
        if limit_per_category is not None:
            selected = questions[:limit_per_category]
        else:
            selected = questions  # Use all questions
        final_dataset.extend(selected)
        print(f"Loaded {len(selected)} for category {cat}")

    random.shuffle(final_dataset)
    print(f"Total dataset: {len(final_dataset)}")
    return final_dataset


def get_existing_progress(filepath: str) -> Set[str]:
    finished = set()
    if not os.path.exists(filepath):
        return finished
    with open(filepath, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
                sig = f"{d['model_id']}|{d['method']}|{d['k']}|{d['question']}"
                finished.add(sig)
            except:
                continue
    return finished


# ==========================================
# MAIN
# ==========================================


def main():
    # ----------------------------
    # 1. Build Engines
    # ----------------------------
    print("\n=== Initializing Engines ===")

    text_engine = get_cached_engine(
        "text_rag_optimal",
        lambda: TextRAGEngine(passages=load_passages(chunk_size=800, overlap=100)),
    )
    # graph_engine = get_cached_engine("graph_rag_v1", GraphRAGEngine)
    graph_engine = GraphRAGEngine()
    baseline_engine = BaselineEngine()
    retrieval_cache = RetrievalCache()
    hybrid_engine = HybridRAGEngine(text_engine, graph_engine)

    # ----------------------------
    # 2. Dual-Judge Consensus (Bias Mitigation)
    # ----------------------------
    # Using heterogeneous panel to mitigate self-preference bias
    # See: Zheng et al. (2023) - "Judging LLM-as-a-Judge"
    print("\n=== Initializing Dual-Judge Panel ===")
    judge_a_llm = TogetherFactDeductorV3(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    )
    judge_b_llm = TogetherFactDeductorV3(
        model="Qwen/Qwen2.5-72B-Instruct-Turbo"
    )
    judge = DualJudge(judge_a_llm, judge_b_llm)
    print(f"  Judge A: {judge.model_a}")
    print(f"  Judge B: {judge.model_b}")

    # ----------------------------
    # 3. Dataset
    # ----------------------------
    dataset = load_evaluation_dataset(INPUT_DATA_FILE)
    # dataset = dataset

    if not dataset:
        return

    finished_runs = get_existing_progress(OUTPUT_FILE)

    # ----------------------------
    # 4. Experiment Loop (Parallelized)
    # ----------------------------
    print(f"\n=== Starting Experiment (Workers: {MAX_WORKERS}) ===")

    def process_question(item, cat_name, model_id, method_name, engine, k, generator, judge):
        """Process a single question - thread-safe worker function."""
        question = item["question"]
        target = item.get("answer")
        facts = item.get("facts", item.get("key_facts", []))

        sig = f"{model_id}|{method_name}|{k}|{question}"

        # Check if already done (thread-safe read)
        with finished_lock:
            if sig in finished_runs:
                return None

        # -------------------------
        # A. RETRIEVAL
        # -------------------------
        cached = retrieval_cache.get(method_name, k, question)
        if cached:
            context = cached["context"]
            retrieval_latency = cached["latency"]
        else:
            t0 = time.time()
            try:
                context = engine.retrieve(question, k=k)
            except Exception:
                context = ""
            retrieval_latency = time.time() - t0
            retrieval_cache.save(
                method_name, k, question, context, retrieval_latency
            )

        # Retrieval tokens
        ctx_tokens = count_tokens(context)
        retrieval_input_tokens = ctx_tokens + count_tokens(question)

        # -------------------------
        # B. GENERATION
        # -------------------------
        t0 = time.time()
        result = generator.deduce_facts(question, context, lang="en")
        generation_latency = time.time() - t0

        prediction = result.get("answer", "No answer.")
        usage = result.get("usage", {})

        # -------------------------
        # TOKEN COUNTING
        # -------------------------
        system_msg = generator._build_system_prompt(lang="en")
        user_msg = generator._build_user_prompt(question, context)
        full_prompt = system_msg + "\n" + user_msg

        model_input_tokens = count_tokens(full_prompt)

        if usage.get("output_tokens", 0) > 0:
            model_output_tokens = usage["output_tokens"]
        else:
            model_output_tokens = count_tokens(prediction)

        # -------------------------
        # COST
        # -------------------------
        total_in = retrieval_input_tokens + model_input_tokens
        total_out = model_output_tokens
        cost = compute_cost(cat_name, total_in, total_out)

        gen_in = model_input_tokens
        gen_out = model_output_tokens

        # -------------------------
        # D. JUDGING
        # -------------------------
        print('Judging response...', prediction)
        try:
            scores = judge.judge_complex(
                question,
                {"answer": target, "facts": facts},
                prediction,
                context,
            )
        except Exception:
            return None

        # -------------------------
        # E. BUILD ROW
        # -------------------------
        row = {
            "model_category": cat_name,
            "model_id": model_id,
            "method": method_name,
            "k": k,
            "q_category": item.get("category"),
            "question": question,
            "target": target,
            "prediction": prediction,
            "context": context,
            # Tokens
            "context_tokens": ctx_tokens,
            "gen_input_tokens": gen_in,
            "gen_output_tokens": gen_out,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            # Latency
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "end_to_end_latency": retrieval_latency + generation_latency,
            # Cost
            "est_cost_usd": cost,
            # Dual-Judge Consensus Scores
            "score": scores["correctness"],
            "faithfulness": scores["faithfulness"],
            # Individual Judge Scores (for bias analysis)
            "judge_a_correctness": scores.get("judge_a_correctness", scores["correctness"]),
            "judge_b_correctness": scores.get("judge_b_correctness", scores["correctness"]),
            "judge_a_faithfulness": scores.get("judge_a_faithfulness", scores["faithfulness"]),
            "judge_b_faithfulness": scores.get("judge_b_faithfulness", scores["faithfulness"]),
        }

        # Thread-safe write
        with write_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(row) + "\n")

        # Thread-safe update
        with finished_lock:
            finished_runs.add(sig)

        return row

    # Main experiment loop
    for cat_name, model_id in MODELS.items():
        print(f"\n🔹 Model: {cat_name} → {model_id}")

        generator = HybridFactDeductor(model_name=model_id)

        methods = {
            "baseline": baseline_engine,
            "text_rag": text_engine,
            "graph_rag": graph_engine,
            "hybrid_rag": hybrid_engine,
        }

        for method_name, engine in methods.items():
            k_values = K_SCHEDULE.get(method_name, [0])

            for k in k_values:
                # Filter to only pending items
                pending_items = []
                for item in dataset:
                    sig = f"{model_id}|{method_name}|{k}|{item['question']}"
                    if sig not in finished_runs:
                        pending_items.append(item)

                if not pending_items:
                    print(f"   → {method_name} (k={k}) - All done ✓")
                    continue

                print(f"   → {method_name} (k={k}) - {len(pending_items)} pending")

                # Process questions in parallel
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(
                            process_question,
                            item, cat_name, model_id, method_name, engine, k, generator, judge
                        ): item
                        for item in pending_items
                    }

                    # Progress bar with tqdm
                    completed = 0
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"      {method_name} k={k}", leave=False):
                        try:
                            row = future.result()
                            if row:
                                completed += 1
                                # Brief inline status
                                ja = row.get("judge_a_correctness", row["score"])
                                jb = row.get("judge_b_correctness", row["score"])
                                tqdm.write(
                                    f"      ✔ Score={row['score']:.2f} (A:{ja:.2f}, B:{jb:.2f}) "
                                    f"| {row['q_category']}"
                                )
                        except Exception as e:
                            tqdm.write(f"      ✗ Error: {e}")

                print(f"      Completed: {completed}/{len(pending_items)}")

    print("\n✅ Experiment complete.")


if __name__ == "__main__":
    main()
