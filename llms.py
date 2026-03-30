import os
import json
import pickle
import random
import re
from typing import Callable, Dict, Any, List, Optional, Set
from together import Together
from dataclasses import dataclass

from transformers import AutoTokenizer

from config import CACHE_DIR
import google.generativeai as genai


# ==========================================
# DATA CLASSES
# ==========================================


@dataclass
class QnAWithMeta:
    question: str
    answer: str
    rationale: str
    # Optional fields for few-shot examples
    expected_terms: List[str] = None
    evidence_spans: List[Dict[str, str]] = None
    difficulty: str = "hard"


# ==========================================
# HELPER FUNCTIONS
# ==========================================


def extract_json_obj(text: str) -> Optional[str]:
    """
    Robustly finds the first JSON object in a string using Regex.
    Useful because LLMs sometimes add "Here is the JSON:" prefixes.
    """
    if not text:
        return None
    # Look for the first outer bracket pair { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Attempts to parse JSON, returns None if failed."""
    try:
        return json.loads(text)
    except Exception:
        return None


# ==========================================
# THE FACT DEDUCTOR (The "Inductor")
# ==========================================


class TogetherFactDeductorV3:
    """
    A strictly fact-grounded answer generator.
    It takes context (from Graph or Text) and produces a JSON answer.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
    ):
        # Fallback to env variable if no key provided
        key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise ValueError("TOGETHER_API_KEY is missing. Check your .env file.")

        self.client = Together(api_key=key)

        self.model = model

    # ---------------------------------------------------------
    # 1. SYSTEM PROMPT: SETS THE BEHAVIOR
    # ---------------------------------------------------------

    def _build_system_prompt(self, lang: str = "en") -> str:
        return (
            "You are an expert on the Icelandic Horse. Your goal is to answer exam questions accurately.\n\n"
            "HIERARCHY OF TRUTH:\n"
            "1. **Primary Source:** The provided 'Context' (Evidence). You must prioritize this information above all else.\n"
            "2. **Secondary Source:** Your internal training knowledge. Use this ONLY if the Context is missing, incomplete, or silent on the topic.\n\n"
            "OUTPUT RULES:\n"
            "- Be concise and direct.\n"
            "- Your output must be valid JSON with keys: 'answer' and 'rationale'."
        )

    def _build_user_prompt(self, question: str, facts_text: str, fewshot=None) -> str:
        if not facts_text or len(facts_text.strip()) < 5:
            context_display = "(No context provided. Rely on internal knowledge.)"
        else:
            context_display = facts_text

        return (
            f"=== INSTRUCTIONS ===\n"
            f"Answer the question using the Hierarchy of Truth (Context > Internal Memory).\n"
            f'Output Format: {{"answer": "...", "rationale": "..."}}\n\n'
            f"=== CONTEXT ===\n"
            f"{context_display}\n\n"
            f"=== QUESTION ===\n"
            f"{question}\n"
        )

    # ---------------------------------------------------------
    # 3. MAIN EXECUTION METHOD
    # ---------------------------------------------------------
    def deduce_facts(
        self,
        question: str,
        facts: str,
        fewshot: List[QnAWithMeta] = None,
        lang: str = "en",
    ) -> Dict[str, Any]:

        system_msg = self._build_system_prompt(lang)
        user_msg = self._build_user_prompt(question, facts, fewshot)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=3000,
                top_p=1,
                stop=["<|eot_id|>", "```"],
            )

            raw_text = response.choices[0].message.content or ""

            # --- Token usage extraction ---
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

            # --- JSON extraction ---
            extracted_json = extract_json_obj(raw_text)
            if not extracted_json:
                return {
                    "answer": raw_text.strip(),
                    "rationale": "JSON parse failure.",
                    "error": True,
                    "usage": usage,
                }

            parsed = safe_json_loads(extracted_json)
            if not parsed:
                return {
                    "answer": raw_text,
                    "rationale": "Invalid JSON.",
                    "error": True,
                    "usage": usage,
                }

            return {
                "answer": parsed.get("answer", ""),
                "rationale": parsed.get("rationale", ""),
                "usage": usage,  # << THE FIX
            }

        except Exception as e:
            return {
                "answer": "",
                "rationale": str(e),
                "error": True,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }


import requests


class LLMJSON:
    """
    Provider-agnostic helper that ALWAYS returns a parsed JSON object (dict or list).
    Supports: OpenAI, Anthropic, Gemini, Ollama-compatible (llama), xAI Grok, DeepSeek, Alibaba Qwen.
    Pick via env: LLM_PROVIDER=openai|anthropic|gemini|llama|grok|deepseek|alibaba|qwen
    Optionally override per instance with provider_hint/model_hint.
    """

    def __init__(
        self,
        provider_hint: Optional[str] = None,
        model_hint: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        # allow per-call override
        self.provider = (provider_hint or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.model = model_hint or os.getenv("LLM_MODEL", "")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "300"))

        if self.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-5")

        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = self.model or os.getenv(
                "CLAUDE_MODEL", "claude-sonnet-4-5-20250929"
            )

        elif self.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.client = genai.GenerativeModel(
                self.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )

        elif self.provider == "llama":
            # Ollama-compatible local server
            self.requests = requests
            self.base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.model = self.model or os.getenv(
                "OLLAMA_MODEL", "qwen3-coder:480b-cloud"
            )

        elif self.provider == "qwen":
            self.requests = requests
            self.base = os.getenv("QWEN_HOST", "http://localhost:11434")
            self.model = self.model or os.getenv("QWEN_MODEL", "qwen-1.0")

        elif self.provider == "grok":
            from openai import OpenAI

            api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GROK_API_KEY / XAI_API_KEY")
            self.client = OpenAI(
                api_key=api_key, base_url="https://api.x.ai/v1"  # xAI endpoint
            )
            self.model = self.model or os.getenv("GROK_MODEL", "grok-4-fast-reasoning")

        elif self.provider == "deepseek":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
            self.model = self.model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        elif self.provider == "alibaba":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.environ.get("ALIBABA_API_KEY")
                or os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
            # Examples: qwen-plus, qwen-turbo, qwen-max, qwen-coder-plus
            self.model = self.model or os.getenv("ALIBABA_MODEL", "qwen-plus")

        else:
            raise RuntimeError(f"Unknown LLM_PROVIDER={self.provider}")

    def __call__(
        self, system: str, prompt: str, response_schema: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Returns parsed JSON (dict or list). Retries if parsing fails.
        """
        text = self._raw_call(system, prompt, response_schema)
        obj = self._force_json(text)

        tries = 3
        while obj is None and tries > 0:
            tries -= 1
            # Print the first 1000 chars (the original code sliced the tail by mistake)
            print(text[:1000])
            print(f"Obj is not a valid JSON, retrying... ({3 - tries}/ 3)")

            # IMPORTANT: do not concatenate; request a fresh strict-JSON reply
            text = self._raw_call(
                system,
                prompt + "\n\nReply with STRICT JSON only. No prose. No code fences.",
                response_schema,
            )
            obj = self._force_json(text)

        return obj

    # ---- providers ----
    def _raw_call(
        self, system: str, user: str, response_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        if response_schema is not None:
            # Append schema to user prompt as fallback for all providers
            user += (
                "\n\nOutput strictly valid JSON matching this schema:\n"
                + json.dumps(response_schema, indent=2)
            )

        if self.provider == "openai":
            r = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # temperature=0
            )
            return r.output_text

        if self.provider == "anthropic":
            r = self.client.messages.create(
                timeout=self.timeout,
                model=self.model,
                temperature=0,
                max_tokens=30000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return r.content[0].text

        if self.provider == "gemini":
            try:
                r = self.client.generate_content(f"{system}\n\n{user}")
            except Exception as e:
                print(f"Gemini API error: {e}")
                return ""
            return r.text or ""

        if self.provider == "llama":
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
            r = self.requests.post(f"{self.base}/api/chat", json=payload, timeout=450)
            js = r.json()
            msg = js.get("message", {})
            return msg.get("content", "")

        if self.provider == "grok":
            # Rough headroom estimator for huge contexts
            def estimate_tokens(text: str) -> int:
                return len(text) // 4 + 100

            full_prompt = f"{system}\n\n{user}"
            prompt_tokens_est = estimate_tokens(full_prompt)
            context_limit = 2_000_000
            max_output_tokens = min(
                1_000_000, context_limit - prompt_tokens_est - 5_000
            )

            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=max_output_tokens,
                temperature=0,
                timeout=600,
                stream=False,
            )
            # optional: debug log
            try:
                os.makedirs("outputs", exist_ok=True)
                with open(
                    "outputs/grok_streaming_debug.jsonl", "a", encoding="utf-8"
                ) as f:
                    json.dump(response.output_text, f)
                    f.write("\n")
            except Exception:
                pass
            return response.output_text

        if self.provider == "deepseek":
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=False,
                temperature=1.0,
            )
            try:
                return r.choices[0].message.content.strip()
            except Exception:
                return ""

        raise RuntimeError("Unsupported provider")

    # ---------- Robust JSON parsing helpers ----------
    @staticmethod
    def _strip_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # Remove all backticks and leading language tag like ```json
            s = s.strip("`").lstrip()
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
        return s

    @staticmethod
    def _remove_trailing_commas(s: str) -> str:
        # Remove commas before } or ] (common LLM hiccup)
        return re.sub(r",(\s*[}\]])", r"\1", s)

    @staticmethod
    def _extract_top_level_json(s: str) -> Optional[str]:
        # Try array first, then object
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = s.find(open_ch)
            if start == -1:
                continue
            depth = 0
            for i, ch in enumerate(s[start:], start=start):
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        return None

    @classmethod
    def _force_json(cls, text: str) -> Optional[Any]:
        if not text:
            return None
        s = cls._strip_fences(text)

        # 1) Try whole string
        try:
            return json.loads(s)
        except Exception:
            pass

        # 2) Extract first complete top-level JSON block (array or object)
        block = cls._extract_top_level_json(s)
        if block:
            try:
                return json.loads(block)
            except Exception:
                # 3) Try after removing trailing commas
                try:
                    return json.loads(cls._remove_trailing_commas(block))
                except Exception:
                    return None

        return None


# ==========================================
# EXAMPLE USAGE (For testing)
# ==========================================
if __name__ == "__main__":
    # Test 1: Graph Context (Bullets)
    graph_context = (
        "- (Tension, causes, Pacy Tölt)\n"
        "- (Pacy Tölt, is_a, Gait Fault)\n"
        "- (Rider, should_fix, Tension)"
    )

    # Test 2: Text Context (Paragraphs)
    text_context = (
        "Pacy tölt is often a result of mental or physical tension in the horse. "
        "To correct this, the rider must first address the underlying tension."
    )

    deductor = TogetherFactDeductorV3(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    )

    q = "What causes pacy tölt?"

    print("--- Graph RAG Output ---")
    print(deductor.deduce_facts(q, graph_context))

    print("\n--- Text RAG Output ---")
    print(deductor.deduce_facts(q, text_context))


# ==========================================
# HYBRID DEDUCTOR (The Fix)
# ==========================================
class HybridFactDeductor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_gemini = "gemini" in model_name.lower()

        if self.is_gemini:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            print(self.model_name)
            self.gemini_model = genai.GenerativeModel(self.model_name)
        else:
            self.client = TogetherFactDeductorV3(model=model_name)

    # --- INTERNAL HELPERS (Moved inside to avoid ImportErrors) ---
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Robustly extracts JSON from text."""
        if not text:
            return {}
        try:
            # 1. Try pure parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try regex extraction
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 3. Fallback: Return text as answer if parsing fails entirely
        return {"answer": text.strip(), "rationale": "Failed to parse JSON."}

    def _build_system_prompt(self, lang: str = "en") -> str:
        return (
            "You are an expert on the Icelandic Horse. Your goal is to answer exam questions accurately.\n\n"
            "HIERARCHY OF TRUTH:\n"
            "1. **Primary Source:** The provided 'Context' (Evidence). You must prioritize this information above all else.\n"
            "2. **Secondary Source:** Your internal training knowledge. Use this ONLY if the Context is missing, incomplete, or silent on the topic.\n\n"
            "OUTPUT RULES:\n"
            "- Be concise and direct.\n"
            "- Your output must be valid JSON with keys: 'answer' and 'rationale'."
        )

    def _build_user_prompt(self, question: str, facts_text: str, fewshot=None) -> str:
        if not facts_text or len(facts_text.strip()) < 5:
            context_display = "(No context provided. Rely on internal knowledge.)"
        else:
            context_display = facts_text

        return (
            f"=== INSTRUCTIONS ===\n"
            f"Answer the question using the Hierarchy of Truth (Context > Internal Memory).\n"
            f'Output Format: {{"answer": "...", "rationale": "..."}}\n\n'
            f"=== CONTEXT ===\n"
            f"{context_display}\n\n"
            f"=== QUESTION ===\n"
            f"{question}\n"
        )

    def deduce_facts(
        self, question: str, facts: str, fewshot=None, lang="en"
    ) -> Dict[str, Any]:
        sys_msg = self._build_system_prompt(lang)
        user_msg = self._build_user_prompt(question, facts, fewshot)

        if self.is_gemini:
            try:
                prompt = sys_msg + "\n\n" + user_msg

                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=4192,
                    ),
                )

                raw_text = response.text

                # --- Extract token usage ---
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
                print("Gemini tings")
                print("usage:", usage)
                print(usage)

                parsed = HybridFactDeductor._extract_json(raw_text)

                return {
                    "answer": parsed.get("answer", ""),
                    "rationale": parsed.get("rationale", ""),
                    "usage": usage,
                }

            except Exception as e:
                return {
                    "answer": "",
                    "rationale": str(e),
                    "error": True,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }

        # --- TOGETHER / OPENAI BRANCH ---
        else:
            try:
                # Use the underlying TogetherFactDeductorV3's method directly
                result = self.client.deduce_facts(question, facts, fewshot, lang)

                # Ensure 'usage' always exists
                usage = result.get(
                    "usage",
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                )

                return {
                    "answer": result.get("answer", ""),
                    "rationale": result.get("rationale", ""),
                    "usage": usage,
                }

            except Exception as e:
                print(f"[Error] Together/OpenAI Exception: {e}")
                return {
                    "answer": "",
                    "rationale": str(e),
                    "error": True,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }


# ==========================================
# CACHING & UTILS
# ==========================================


class RetrievalCache:
    def __init__(self, filepath="retrieval_cache.jsonl"):
        self.filepath = filepath
        self.cache = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        print(f"📦 Loading Retrieval Cache from {self.filepath}...")
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    self.cache[data["key"]] = data["val"]
                except json.JSONDecodeError:
                    continue
        print(f"   -> {len(self.cache)} entries loaded.")

    def get(self, method, k, question):
        return self.cache.get(f"{method}|{k}|{question}")

    def save(self, method, k, question, context, latency):
        key = f"{method}|{k}|{question}"
        entry = {"context": context, "latency": latency}
        self.cache[key] = entry
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "val": entry}) + "\n")


print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def get_cached_engine(engine_name: str, factory_func: Callable):
    os.makedirs(CACHE_DIR, exist_ok=True)
    file_path = os.path.join(CACHE_DIR, f"{engine_name}.pkl")
    if os.path.exists(file_path):
        print(f"⚡ Loading {engine_name} from cache...")
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            print(f"⚠️ Cache corrupted. Rebuilding...")

    print(f"🔨 Building {engine_name}...")
    engine_instance = factory_func()
    with open(file_path, "wb") as f:
        pickle.dump(engine_instance, f)
    return engine_instance


def load_evaluation_dataset(filepath: str) -> List[Dict[str, Any]]:
    dataset = []
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {filepath} not found.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                dataset.append(json.loads(line))
            except:
                continue
    random.seed(42)
    random.shuffle(dataset)
    return dataset[:4]  # Debug limit


def get_existing_progress(filepath: str) -> Set[str]:
    finished = set()
    if not os.path.exists(filepath):
        return finished
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                sig = f"{data['model_id']}|{data['method']}|{data['k']}|{data['question']}"
                finished.add(sig)
            except:
                continue
    return finished
