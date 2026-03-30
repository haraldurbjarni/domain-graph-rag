from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from tqdm import tqdm
from utils.toc_utils import get_hardcoded_toc
from utils.utils import Passage

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

import re
from typing import Iterator
import fitz

_ABBR_EN = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "al.",
    "fig.",
    "no.",
    "vol.",
}
_ABBR_IS = {
    "þ.e.",
    "t.d.",
    "o.s.frv.",
    "o.fl.",
    "bls.",
    "sbr.",
    "þ.m.t.",
    "t.a.m.",
}

# Simple unicode-aware ellipsis and sentence enders
_SENT_END_RE = re.compile(r"([\.!?…]+)([\)\"»’”\]]*)\s+", re.UNICODE)


def _build_chapter_index(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: chapters = [{'num','name','page','subchapters': {title -> page}}, ...]
    Output: list of chapter segments with:
      - start (inclusive), end (inclusive)
      - subs: Dict[int page -> List[str titles]]  (note: multiple titles per page!)
    """
    # Compute [start,end] for each chapter
    out = []
    for i, ch in enumerate(chapters):
        start = ch.get("page") or ch.get("start") or 1
        next_start = chapters[i + 1].get("page") if i + 1 < len(chapters) else math.inf
        end = (next_start - 1) if next_start is not None else math.inf

        # Collect subchapters by page -> list[str]
        subs_by_page: Dict[int, List[str]] = {}
        for title, p in (ch.get("subchapters") or {}).items():
            subs_by_page.setdefault(int(p), []).append(title)

        out.append(
            {
                "num": ch["num"],
                "name": ch["name"],
                "start": int(start),
                "end": int(end) if end != math.inf else math.inf,
                "subs": subs_by_page,
            }
        )
    return out


def _protect_abbreviations(text: str, abbr: set) -> Tuple[str, Dict[str, str]]:
    """
    Replace dots inside known abbreviations with a placeholder so we don't split on them.
    Returns (protected_text, mapping) so we can restore afterwards.
    """
    mapping = {}
    protected = text
    # Sort by length to replace longer first
    for a in sorted(abbr, key=len, reverse=True):
        token = a.replace(".", "§")  # placeholder
        if a in protected:
            protected = protected.replace(a, token)
            mapping[token] = a
    return protected, mapping


def _restore_abbreviations(text: str, mapping: Dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


def load_passages(base_path: str = "data/pdfs/") -> List["Passage"]:
    print("Parsing all PDF documents...")
    en_sentences = []
    offset_map_en = {0: 0, 1: 0, 2: 2, 3: 2, 4: 2}  # Adjust as
    for idx in tqdm(range(0, 5), desc="Parsing Books"):
        ride_pdf = base_path + f"Riding_Levels_{idx+1}.searchable.pdf"
        if not os.path.isfile(ride_pdf):
            print(f"Skipping level {idx+1}: missing file.")
            continue
        # English
        for s in iter_sentences_en(
            ride_pdf, idx + 1, get_hardcoded_toc, offset=offset_map_en[idx]
        ):
            en_sentences.append({"text": s.text, "lang": s.lang, "source": s.source})

    all_sentences = en_sentences
    print(f"Total sentences loaded: {len(all_sentences)}")
    print(all_sentences[20:30])  # sanity peek
    passages = [Passage(**s) for s in all_sentences]
    return passages


def sentence_tokenize(text: str, lang: str) -> list:
    """
    Heuristic sentence splitter that handles EN/IS abbreviations and unicode punctuation.
    - Keeps end punctuation with the sentence.
    - Filters very short fragments.
    """
    if not text:
        return []

    lang = (lang or "en").lower()
    abbr = _ABBR_IS if lang.startswith("is") else _ABBR_EN

    # Normalize spacing
    t = re.sub(r"[ \t]+", " ", text.strip())

    # Protect abbreviations
    t, mapping = _protect_abbreviations(t, abbr)

    # Split on sentence enders while keeping punctuation with the left side
    parts = []
    start = 0
    for m in _SENT_END_RE.finditer(t):
        end = m.end(1) + (len(m.group(2)) if m.group(2) else 0)
        seg = t[start:end].strip()
        if seg:
            parts.append(seg)
        start = m.end()  # after trailing spaces

    # Tail (after the last match)
    tail = t[start:].strip()
    if tail:
        parts.append(tail)

    # Restore abbreviations and clean
    out = []
    for s in parts:
        s = _restore_abbreviations(s, mapping)
        s = s.strip()
        # Drop diagram placeholders or ultra-short noise
        if s and s != "[Diagram or image-heavy page]" and len(s) >= 2:
            out.append(s)
    return out


def _format_source(page_rec: Dict[str, Any]) -> str:
    """
    Build a compact, human-friendly source string from your page metadata.
    page_rec is the dict yielded by iter_passages_is/en (book/page/logical_page/chapter/subchapter/text).
    """
    book = page_rec.get("book") or ""
    lp = page_rec.get("logical_page") or page_rec.get("page") or ""
    cn = page_rec.get("chapter_num")
    cname = page_rec.get("chapter_name") or ""
    sname = page_rec.get("subchapter_name") or ""
    bits = []
    if book:
        bits.append(book)
    if lp:
        bits.append(f"p.{lp}")
    if cn is not None and cn != 0:
        if cname:
            bits.append(f"ch.{cn}: {cname}")
        else:
            bits.append(f"ch.{cn}")
    if sname:
        bits.append(f"sub: {sname}")
    return " | ".join(bits)


def iter_sentences_from_pages(
    page_iter: Iterable[Dict[str, Any]], lang: str
) -> Iterator[Passage]:
    """
    Takes your existing page iterator (iter_passages_is/en) and yields sentence-level Passages.
    """
    for rec in page_iter:
        txt = rec.get("text", "") or ""
        if not txt or txt == "[Diagram or image-heavy page]":
            continue
        for s in sentence_tokenize(txt, lang):
            yield Passage(text=s, lang=lang, source=_format_source(rec))


def _assign_chapter_only(ch_index, logical_page):
    """Return (chapter_num, chapter_name, sub_titles_on_this_page:list[str])."""
    if ch_index and logical_page < ch_index[0]["start"]:
        return 0, "Intro", []
    for ch in ch_index:
        if ch["start"] <= logical_page <= ch["end"]:
            titles = ch["subs"].get(logical_page, [])
            return ch["num"], ch["name"], titles
    if ch_index and logical_page > ch_index[-1]["end"]:
        last = ch_index[-1]
        return last["num"], last["name"], []
    return 0, "Intro", []


def iter_passages_en(pdf_path, level, get_toc_func, offset=0, skip_first=0):
    toc_all = get_toc_func()
    if level not in toc_all:
        raise KeyError(f"No English TOC for level {level}.")
    ch_index = _build_chapter_index(toc_all[level])

    with fitz.open(pdf_path) as doc:
        title = doc.metadata.get("title") or os.path.basename(pdf_path)
        current_chapter_num = None
        current_sub = None

        for pdf_page, page in enumerate(doc, start=1):
            if pdf_page <= skip_first:
                continue
            logical_page = pdf_page - offset

            cnum, cname, sub_titles = _assign_chapter_only(ch_index, logical_page)

            if cnum != current_chapter_num:
                current_chapter_num = cnum
                current_sub = None

            if sub_titles:
                joined_for_this_page = " | ".join(sub_titles)
                current_sub = sub_titles[-1]  # carry this forward
                sname_to_emit = joined_for_this_page
            else:
                sname_to_emit = current_sub

            text = page.get_text("text").strip()
            yield {
                "book": title,
                "page": pdf_page,
                "logical_page": logical_page,
                "chapter_num": cnum,
                "chapter_name": cname,
                "subchapter_name": sname_to_emit,
                "text": text or "[Diagram or image-heavy page]",
            }


def iter_sentences_en(pdf_path, level, get_toc_func, offset=0, skip_first=0):
    pages = iter_passages_en(
        pdf_path, level, get_toc_func, offset=offset, skip_first=skip_first
    )
    yield from iter_sentences_from_pages(pages, lang="en")


@dataclass
class Passage:
    """
    Represents a chunk of text ready for embedding or LLM consumption.
    """

    text: str
    source: str  # Filename or Book Title
    chunk_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# CORE: RECURSIVE CHUNKING LOGIC
# ==========================================


def recursive_chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Splits text into chunks of ~chunk_size characters.
    Respects natural boundaries (paragraphs > sentences > words) and adds overlap.

    Args:
        text: Full document text.
        chunk_size: Target characters per chunk.
        overlap: Number of characters to repeat at the start of the next chunk.
        separators: List of separators to try in order.
    """
    if not text:
        return []

    if separators is None:
        # Hierarchy: Double Newline (Para) -> Newline -> Period (Sentence) -> Space -> Character
        separators = ["\n\n", "\n", ". ", " ", ""]

    # 1. Base Case: If text fits, return it
    if len(text) <= chunk_size:
        return [text]

    # 2. Iterative Splitting
    # Find the best separator that works for this block
    selected_sep = separators[-1]  # Default to char split
    for sep in separators:
        if sep == "":
            continue
        # If this separator actually splits the text, let's see if the chunks are reasonable
        if sep in text:
            # We want to check if splitting by this sep generally gives us pieces smaller than chunk_size
            # This is a heuristic. For strictness, we just pick the highest level separator present.
            selected_sep = sep
            break

    # Split text
    if selected_sep == "":
        splits = list(text)  # Character split
    else:
        splits = text.split(selected_sep)

    # 3. Merge splits into chunks with overlap
    final_chunks = []
    current_chunk = []
    current_len = 0

    # We iterate through the natural splits (e.g., paragraphs)
    for i, split in enumerate(splits):
        # Restore separator length unless it's the last one
        sep_len = len(selected_sep) if i < len(splits) - 1 else 0
        split_len = len(split) + sep_len

        # If adding this split exceeds chunk size AND we have something in buffer
        if current_len + split_len > chunk_size and current_chunk:
            # Join and save current
            text_chunk = selected_sep.join(current_chunk)
            final_chunks.append(text_chunk)

            # --- OVERLAP LOGIC ---
            # Keep the tail of the previous chunk to start the new one
            overlap_buffer = []
            overlap_acc = 0

            # Walk backwards
            for prev_split in reversed(current_chunk):
                if overlap_acc < overlap:
                    overlap_buffer.insert(0, prev_split)
                    overlap_acc += len(prev_split) + len(selected_sep)
                else:
                    break

            current_chunk = overlap_buffer + [split]
            current_len = overlap_acc + split_len

        else:
            current_chunk.append(split)
            current_len += split_len

    # Add leftover
    if current_chunk:
        final_chunks.append(selected_sep.join(current_chunk))

    return final_chunks


# ==========================================
# FILE LOADING
# ==========================================


def clean_text(text: str) -> str:
    """Basic cleaning to remove header/footer noise and excessive whitespace."""
    # Replace weird whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove page numbers (simple heuristic: digit at start/end of line)
    # text = re.sub(r'^\d+\s|\s\d+$', '', text)
    return text.strip()


def load_pdf_text(filepath: str) -> str:
    """Extracts text from a single PDF using pypdf."""
    if PdfReader is None:
        raise ImportError("pypdf is not installed. Run `pip install pypdf`.")

    reader = PdfReader(filepath)
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)
    return "\n\n".join(full_text)


def load_passages(
    source_dir: str = "data/pdfs", chunk_size: int = 1000, overlap: int = 200
) -> List[Passage]:
    """
    Main entry point. Loads all PDFs in directory, chunks them, and returns objects.

    Args:
        chunk_size:
            - Use 1000-1500 for GENERATING questions (need context).
            - Use 500-800 for Text-RAG RETRIEVAL (need precision).
    """
    passages = []
    path = Path(source_dir)

    if not path.exists():
        print(
            f"Warning: Directory {source_dir} does not exist. Returning empty passages."
        )
        return []

    print(f"Loading PDFs from {source_dir}...")
    files = list(path.glob("*.pdf"))

    for file_path in files:
        try:
            # 1. Extract Raw Text
            raw_text = load_pdf_text(str(file_path))
            cleaned_text = clean_text(raw_text)

            # 2. Apply Recursive Chunking
            chunks = recursive_chunk_text(
                cleaned_text, chunk_size=chunk_size, overlap=overlap
            )

            # 3. Create Objects
            for i, chunk_text in enumerate(chunks):
                passages.append(
                    Passage(
                        text=chunk_text,
                        source=file_path.name,
                        chunk_id=i,
                        metadata={"len": len(chunk_text)},
                    )
                )

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print(f"Loaded {len(passages)} passages from {len(files)} documents.")
    return passages
