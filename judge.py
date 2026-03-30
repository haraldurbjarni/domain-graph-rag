import json
import re
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor


# ==========================================
# SINGLE JUDGE (Original Implementation)
# ==========================================


class LLMJudge:
    def __init__(self, llm_client):
        """
        llm_client: Instance of TogetherFactDeductorV3
        """
        self.client = llm_client

    def _parse_json(self, text: str) -> Dict:
        # Robust parsing that handles "Here is the JSON:" preambles
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}
        except:
            return {}

    def _call_llm_grading(self, prompt: str) -> float:
        """
        Helper to execute the grading prompt.
        Returns a float between 0.0 and 1.0.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            # Accessing the inner client directly for raw generation
            response = self.client.client.chat.completions.create(
                model=self.client.model,
                messages=messages,
                temperature=0.0,  # Deterministic grading
                max_tokens=1024,
            )
            raw_text = response.choices[0].message.content
            data = self._parse_json(raw_text)

            # Extract score (1-5) and normalize to 0.0-1.0
            raw_score = float(data.get("score", 0))
            normalized_score = max(0.0, min(1.0, raw_score / 5.0))
            return normalized_score
        except Exception as e:
            print(f"Grading Failed: {e}")
            return 0.0

    def judge_complex(
        self, question: str, gold_data: Dict, prediction: str, context: str
    ) -> Dict[str, float]:
        gold_answer = gold_data.get("answer", "")
        # If 'facts' is a list, join it, otherwise use as string
        facts_raw = gold_data.get("facts", [])
        gold_facts = (
            "\n".join(facts_raw) if isinstance(facts_raw, list) else str(facts_raw)
        )

        # --- 1. ACCURACY PROMPT (G-Eval Style) ---
        accuracy_prompt = f"""
        You are an expert evaluator for an Icelandic Horse examination. 
        Compare the Predicted Answer to the Ground Truth.

        QUESTION: {question}
        
        GROUND TRUTH:
        {gold_answer}
        
        REQUIRED FACTS/KEYWORDS:
        {gold_facts}
        
        PREDICTED ANSWER:
        {prediction}

        Evaluation Steps:
        1. Does the prediction answer the specific question asked?
        2. Does it contain the specific keywords/facts listed above?
        3. Is the reasoning logical and consistent with the Ground Truth?

        Scoring Rubric (1-5):
        5: Perfect. Contains all facts, correct reasoning, and precise terminology.
        4: Good. Missing minor nuance or one keyword, but semantically identical.
        3: Acceptable. Correct answer but vague, lacks specific details/terminology.
        2: Weak. Related information is present, but fails to answer the specific prompt accurately.
        1: Wrong. Factually incorrect or irrelevant.
        
        Return JSON ONLY:
        {{
            "reasoning": "Short explanation of the score...",
            "score": <integer 1-5>
        }}
        """

        # --- 2. FAITHFULNESS PROMPT ---
        faithfulness_prompt = f"""
        Evaluate if the Predicted Answer is grounded in the Context.
        
        CONTEXT:
        {context[:8000]} (truncated if too long)
        
        PREDICTED ANSWER:
        {prediction}
        
        Scoring Rubric (1-5):
        5: Fully supported. Every claim in the answer exists in the Context.
        4: Mostly supported. Minor extrapolations that are common sense.
        3: Mixed. Some claims are supported, but others are hallucinations or external knowledge.
        2: Poorly supported. Answer contradicts context or relies heavily on external info.
        1: Unsupported. Answer ignores context or is pure hallucination.
        
        Return JSON ONLY:
        {{
            "reasoning": "Short explanation...",
            "score": <integer 1-5>
        }}
        """

        acc_score = self._call_llm_grading(accuracy_prompt)
        faith_score = self._call_llm_grading(faithfulness_prompt)

        return {
            "correctness": acc_score,  # This will be e.g., 0.8, 0.6, 1.0
            "faithfulness": faith_score,
        }

    def judge_simple(self, question: str, target: str, prediction: str) -> float:
        # Re-use the complex accuracy logic for simple questions, just without the extra facts
        return self.judge_complex(
            question, {"answer": target, "facts": []}, prediction, ""
        )["correctness"]


# ==========================================
# DUAL-JUDGE CONSENSUS (Bias Mitigation)
# ==========================================


class DualJudge:
    """
    Dual-Judge Consensus mechanism to mitigate Self-Preference Bias.

    Uses a heterogeneous panel of two state-of-the-art models:
    - Judge A: Llama-3.1-70B (meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo)
    - Judge B: Qwen2.5-72B (Qwen/Qwen2.5-72B-Instruct-Turbo)

    By averaging scores from two distinct model families, we reduce variance
    associated with single-model idiosyncrasies and dampen family-specific
    stylistic preferences.

    Reference: Zheng et al. (2023) - "Judging LLM-as-a-Judge"
    """

    # Default judge models
    JUDGE_A_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    JUDGE_B_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"

    def __init__(self, llm_client_a, llm_client_b):
        """
        Args:
            llm_client_a: LLM client for Judge A (Llama family)
            llm_client_b: LLM client for Judge B (Qwen family)
        """
        self.judge_a = LLMJudge(llm_client_a)
        self.judge_b = LLMJudge(llm_client_b)
        self.model_a = getattr(llm_client_a, 'model', self.JUDGE_A_MODEL)
        self.model_b = getattr(llm_client_b, 'model', self.JUDGE_B_MODEL)

    def _average_scores(self, scores_a: Dict[str, float], scores_b: Dict[str, float]) -> Dict[str, float]:
        """Average the scores from both judges."""
        return {
            key: (scores_a.get(key, 0.0) + scores_b.get(key, 0.0)) / 2.0
            for key in set(scores_a.keys()) | set(scores_b.keys())
        }

    def judge_complex(
        self, question: str, gold_data: Dict, prediction: str, context: str
    ) -> Dict[str, Any]:
        """
        Evaluate using dual-judge consensus.

        Returns:
            Dict containing:
            - correctness: Averaged correctness score (0.0-1.0)
            - faithfulness: Averaged faithfulness score (0.0-1.0)
            - judge_a_correctness: Individual score from Judge A
            - judge_b_correctness: Individual score from Judge B
            - judge_a_faithfulness: Individual score from Judge A
            - judge_b_faithfulness: Individual score from Judge B
        """
        # Run both judges in parallel for efficiency
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                self.judge_a.judge_complex, question, gold_data, prediction, context
            )
            future_b = executor.submit(
                self.judge_b.judge_complex, question, gold_data, prediction, context
            )

            scores_a = future_a.result()
            scores_b = future_b.result()

        # Compute consensus (average)
        consensus = self._average_scores(scores_a, scores_b)

        return {
            # Consensus scores (primary output)
            "correctness": consensus.get("correctness", 0.0),
            "faithfulness": consensus.get("faithfulness", 0.0),
            # Individual judge scores (for analysis)
            "judge_a_correctness": scores_a.get("correctness", 0.0),
            "judge_b_correctness": scores_b.get("correctness", 0.0),
            "judge_a_faithfulness": scores_a.get("faithfulness", 0.0),
            "judge_b_faithfulness": scores_b.get("faithfulness", 0.0),
        }

    def judge_simple(self, question: str, target: str, prediction: str) -> float:
        """Simple evaluation using dual-judge consensus."""
        return self.judge_complex(
            question, {"answer": target, "facts": []}, prediction, ""
        )["correctness"]
