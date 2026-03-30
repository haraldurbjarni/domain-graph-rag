import json
import random
import os
import re
import string
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
import dotenv

from llms import LLMJSON
from utils.parse_utils import load_passages

dotenv.load_dotenv()

# --- CONFIG ---
OUTPUT_FILE = "generated_questions_sota.jsonl"
TOTAL_QUESTIONS_NEEDED = 60  # We generate more, then filter
CHUNK_SIZE = 2000

llm = LLMJSON(provider_hint="gemini", model_hint="gemini-3-pro-preview")

# ==========================================
# 1. THE "BLIND" TEACHER PROMPT
# ==========================================


def build_teacher_prompt(context_text: str) -> str:
    return f"""
    You are a Professor of Equine Science creating a final exam for the Icelandic Horse certification.
    
    Below is a specific excerpt (Context) from the course material.
    
    CONTEXT:
    "{context_text[:5000]}"
    
    TASK:
    Generate 1-2 deep, high-quality exam questions based **strictly** on the logic found in this context.
    
    CRITICAL RULES:
    1. **Natural Difficulty:** Do not force complexity if the text is simple. If the text explains a complex cause-and-effect, ask a Causal question. If it lists items, ask an Aggregation question.
    2. **No Meta-Talk:** The question MUST NOT contain phrases like "According to the text", "Based on the excerpt", or "In this passage". It must sound like a general truth question.
    3. **Self-Classification:** After writing the question, you must analyze it and classify it into one of: ['Lookup', 'Multi-hop', 'Causal', 'Aggregation'].
    
    OUTPUT FORMAT (JSON List):
    [
      {{
        "question": "When training the Tölt, how does the rider's seat position influence the engagement of the hindquarters?",
        "answer": "A slight weight shift back allows the horse to lower the croup...",
        "rationale": "The text explains the biomechanical link between seat weight (Factor A) and hindquarter engagement (Factor B).",
        "key_facts": ["Weight shift back lowers croup", "Lowered croup frees shoulders"],
        "category": "Causal", 
        "difficulty": "Hard"
      }}
    ]
    """


# ==========================================
# 2. CONTEXT PREPARATION (The "Mixer")
# ==========================================


def get_keywords(text: str) -> Set[str]:
    translator = str.maketrans("", "", string.punctuation)
    words = text.translate(translator).lower().split()
    stopwords = {
        "the",
        "and",
        "is",
        "of",
        "to",
        "in",
        "a",
        "that",
        "for",
        "with",
        "horse",
        "rider",
        "are",
        "this",
        "from",
        "have",
        "ride",
        "riding",
        "very",
        "much",
        "many",
        "some",
        "time",
        "good",
        "well",
        "should",
        "when",
        "then",
        "will",
        "what",
        "where",
        "training",
        "horses",
        "chapter",
        "page",
        "section",
        "figure",
        "table",
    }
    return {w for w in words if w not in stopwords and len(w) > 4}


def prepare_blind_contexts(
    passages: List[Any], n_local: int, n_global: int
) -> List[Dict]:
    """
    Creates a mixed list of contexts.
    Some are single chunks (Local). Some are keyword-bridged pairs (Global).
    The generator will not know the difference.
    """
    mixed_contexts = []

    # 1. Local Contexts (Single Chunks)
    # We shuffle and pick random single chunks
    indices = list(range(len(passages)))
    random.shuffle(indices)

    print(f"Preparing {n_local} Local contexts...")
    for i in indices[:n_local]:
        mixed_contexts.append(
            {
                "text": passages[i].text,
                "source_type": "local",  # We track this metadata, but don't show the LLM
                "chunk_ids": [passages[i].chunk_id],
            }
        )

    # 2. Global Contexts (Bridged Pairs)
    # We define "Global" as two chunks that share a concept but are physically distant.
    print(f"Preparing {n_global} Global contexts...")
    kw_index = defaultdict(list)
    for i, p in enumerate(passages):
        for k in get_keywords(p.text):
            kw_index[k].append(i)

    valid_kws = [k for k, v in kw_index.items() if 2 <= len(v) < 30]
    random.shuffle(valid_kws)

    pairs_found = 0
    seen_pairs = set()

    for k in valid_kws:
        if pairs_found >= n_global:
            break

        inds = kw_index[k]
        idx_a, idx_b = random.sample(inds, 2)
        p_a, p_b = passages[idx_a], passages[idx_b]

        # Enforce distance (must be from different files or >10 chunks apart)
        is_distant = (p_a.source != p_b.source) or (
            abs(p_a.chunk_id - p_b.chunk_id) > 10
        )

        pair_sig = tuple(sorted((p_a.chunk_id, p_b.chunk_id)))

        if is_distant and pair_sig not in seen_pairs:
            # Combine them into one "Document"
            combined_text = f"{p_a.text}\n\n{p_b.text}"
            mixed_contexts.append(
                {
                    "text": combined_text,
                    "source_type": "global",  # Metadata
                    "chunk_ids": [p_a.chunk_id, p_b.chunk_id],
                }
            )
            seen_pairs.add(pair_sig)
            pairs_found += 1

    return mixed_contexts


# ==========================================
# 3. UTILS
# ==========================================


def clean_question(text: str) -> str:
    # Aggressive regex to remove leakage
    text = re.sub(
        r"(According to|Based on|In) the (text|passage|excerpt|context|documents?),?",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"What does the text say about", "What is", text, flags=re.I)
    text = text.strip()
    return text[0].upper() + text[1:] if text else text


def extract_json(response: Any) -> List[Dict]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        return [response]
    if isinstance(response, str):
        match = re.search(r"\[.*\]", response.strip(), re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return []


# ==========================================
# 4. MAIN
# ==========================================


def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # 1. Load passages
    print("Loading passages...")
    passages = load_passages(chunk_size=CHUNK_SIZE, overlap=200)
    # Filter very short chunks
    valid_passages = [p for p in passages if len(p.text) > 500]

    # 2. Create the Blind Mix
    # We ask for 50/50 split, then we shuffle them
    contexts = prepare_blind_contexts(valid_passages, n_local=40, n_global=40)

    # SHUFFLE IS CRITICAL FOR BIAS REMOVAL
    random.shuffle(contexts)

    print(f"Starting generation on {len(contexts)} mixed contexts...")

    total = 0
    category_counts = defaultdict(int)

    for ctx in tqdm(contexts):
        try:
            # The Prompt doesn't know if it's local or global
            prompt = build_teacher_prompt(ctx["text"])

            resp = llm(system="", prompt=prompt)
            items = extract_json(resp)

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for q in items:
                    q["question"] = clean_question(q.get("question", ""))

                    # We inject the hidden metadata here for your Thesis analysis
                    q["source_type"] = ctx["source_type"]
                    q["gold_context"] = ctx["text"]

                    # Ensure fields exist
                    if "facts" not in q:
                        q["facts"] = q.get("key_facts", [])

                    # Normalize Category
                    cat = q.get("category", "Lookup").title()
                    category_counts[cat] += 1

                    f.write(json.dumps(q) + "\n")
                    total += 1
        except Exception as e:
            pass  # Skip failed generations

    print(f"\n✅ Done. Generated {total} questions.")
    print("Distribution of Self-Classified Categories:")
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
