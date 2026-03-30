from dataclasses import dataclass, field, field
import json
import os
from typing import Any, Dict, Iterator, Iterator, Iterable, List
import unicodedata

import tqdm


# Classes
@dataclass
class QnA:
    question_is: str
    question_en: str
    answer: str
    answer_en: str
    why: str


@dataclass
class QnAWithMeta:
    question: str
    answer: str
    rationale: str
    expected_terms: List[str]
    evidence_spans: List[Dict[str, str]]
    difficulty: str


@dataclass
class Evidence:
    id: int
    text: str
    sentence_index: int
    char_start: int
    char_end: int


@dataclass
class Passage:
    text: str
    lang: str  # "en" or "is"
    source: str  # optional, can be ""


@dataclass
class Fact:
    references: List[int]
    fact: Dict[str, str]  # e.g., {"subject": str, "predicate": str, "object": str}


@dataclass
class Checks:
    uses_multiple_blocks: str
    all_spans_verbatim: str
    answer_supported_by_gold: str


@dataclass
class QuestionData:
    question: str
    answer: str
    rationale: str
    expected_terms: List[str] = field(default_factory=list)
    gold_evidence: List[Evidence] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    query_terms: List[str] = field(default_factory=list)
    query_variants: List[str] = field(default_factory=list)
    difficulty: str = "hard"
    reasoning_type: List[str] = field(default_factory=list)
    checks: Checks = field(default_factory=lambda: Checks("true", "true", "true"))


def normalize_text(text):
    """Removes accents and lowercases text for robust matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    ).lower()


def extract_node_data(raw_node):
    """
    Safely extracts ID, Label, and Name from raw node data
    (handling both Strings and Dictionaries).
    """
    if isinstance(raw_node, str):
        nid = raw_node
        label = "Concept"
        name = raw_node.replace("_", " ").title()
    else:
        nid = raw_node["id"]
        label = raw_node.get("label", "Concept")
        name = raw_node.get("name", nid)
    return nid, label, name


def load_qna_from_json(file_path: str) -> List[QnA]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [QnA(**item) for item in data]


def load_qnas(base_path: str) -> List[QnA]:
    all_qnas = []
    for level in range(1, 6):
        fp = f"{base_path}/qa_level{level}.json"
        if os.path.exists(fp):
            all_qnas.extend(load_qna_from_json(fp))
    return all_qnas


def parse_jsonl_to_question_data() -> List[QuestionData]:
    file_path = "complex_questions/complex_questions.jsonl"
    """
    Reads a JSONL file where each line is a JSON object with 'ind' and 'data' keys.
    Parses the 'data' into QuestionData dataclasses.
    
    Args:
        file_path (str): Path to the JSONL file.
    
    Returns:
        List[QuestionData]: List of parsed QuestionData objects.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If a line can't be parsed (logged, but continues).
        ValueError: If required fields are missing.
    """
    objects = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    obj = json.loads(line)
                    if "data" not in obj:
                        print(
                            f"Warning: Line {line_num} missing 'data' key: {line[:50]}..."
                        )
                        continue

                    data_dict = obj["data"]

                    # Parse sub-dataclasses
                    checks_data = data_dict.get("checks", {})
                    checks = Checks(
                        uses_multiple_blocks=checks_data.get(
                            "uses_multiple_blocks", "true"
                        ),
                        all_spans_verbatim=checks_data.get(
                            "all_spans_verbatim", "true"
                        ),
                        answer_supported_by_gold=checks_data.get(
                            "answer_supported_by_gold", "true"
                        ),
                    )

                    evidence_list = []
                    for ev in data_dict.get("gold_evidence", []):
                        evidence_list.append(
                            Evidence(
                                id=ev.get("id"),
                                text=ev.get("text"),
                                sentence_index=ev.get("sentence_index"),
                                char_start=ev.get("char_start"),
                                char_end=ev.get("char_end"),
                            )
                        )

                    facts_list = []
                    for ft in data_dict.get("facts", []):
                        facts_list.append(
                            Fact(
                                references=ft.get("references", []),
                                fact=ft.get("fact", {}),
                            )
                        )

                    # Main dataclass
                    q_data = QuestionData(
                        question=data_dict.get("question", ""),
                        answer=data_dict.get("answer", ""),
                        rationale=data_dict.get("rationale", ""),
                        expected_terms=data_dict.get("expected_terms", []),
                        gold_evidence=evidence_list,
                        facts=facts_list,
                        query_terms=data_dict.get("query_terms", []),
                        query_variants=data_dict.get("query_variants", []),
                        difficulty=data_dict.get("difficulty", "hard"),
                        reasoning_type=data_dict.get("reasoning_type", []),
                        checks=checks,
                    )

                    objects.append(q_data)

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Error parsing line {line_num}: {e}")
                    print(f"Line content: {line[:100]}...")
                    continue
        print(
            f"Successfully parsed {len(objects)} QuestionData objects from {file_path}."
        )
        return objects
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")
