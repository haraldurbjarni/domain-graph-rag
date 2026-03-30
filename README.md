# End-to-End Graph Retrieval Pipeline for Specialized Domains

Code for the paper presented at the [KG & LLM Workshop](https://kg-llm.github.io/) @ LREC 2026 (Palma de Mallorca):

> **End-to-End Graph Retrieval Pipeline for Specialized Domains**
> Haraldur Bjarni Davidsson, Hazar Harmouch
> Vrije Universiteit Amsterdam, University of Amsterdam

## Overview

This repository contains the full pipeline for constructing a domain-specific **hyper-relational knowledge graph** from instructional text using LLM-assisted extraction, and evaluating it against Text-RAG and Hybrid-RAG baselines on an expert-validated QA benchmark.

The pipeline was applied to the *Icelandic Riding Levels*, a 602-page training corpus for riders of the Icelandic Horse. The source PDFs are not included due to copyright, but the pipeline is **corpus-agnostic** and can be run on any set of PDF documents.

### Key components

| Component | File | Description |
|---|---|---|
| KG Construction | `kg_pipeline.py` | LLM-based extraction of hyper-relational triples with schema-constrained qualifiers, followed by embedding-based entity resolution |
| Text-RAG | `engines.py` | Hybrid dense (`multilingual-e5-base`) + sparse (BM25) retrieval, fused via Reciprocal Rank Fusion, re-ranked with a cross-encoder |
| Graph-RAG | `engines.py` | Seed entity identification via hybrid retrieval, Personalized PageRank expansion with hub penalization, cross-encoder edge re-ranking |
| Hybrid-RAG | `engines.py` | Dynamic combination of Text-RAG and Graph-RAG (k text chunks + 10k graph triples) |
| Evaluation | `run_experiment.py` | Full experiment harness with dual-judge consensus scoring (Llama-3.1-70B + Qwen2.5-72B) |
| Question Generation | `generate_questions.py` | LLM-based QA benchmark synthesis from source chunks |
| LLM Abstraction | `llms.py` | Provider-agnostic LLM interface supporting Together AI, Gemini, OpenAI, Anthropic, and others |

## Setup

### Requirements

- Python >= 3.11
- API keys for at least one LLM provider (see below)

### Installation

```bash
git clone https://github.com/haraldurbjarni/domain-graph-rag.git
cd domain-graph-rag

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for KG construction (extraction uses Gemini)
GEMINI_API_KEY=your_key_here

# Required for evaluation (generation + judging uses Together AI)
TOGETHER_API_KEY=your_key_here

# Optional (for other LLM providers in llms.py)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### 1. Prepare Your Corpus

Place PDF files in `data/pdfs/`. The pipeline extracts text, segments it into ~1,000-character chunks with 200-character overlap, and processes them through the KG construction pipeline.

### 2. Build the Knowledge Graph

```bash
python kg_pipeline.py
```

This will:
1. Parse PDFs into text chunks (`utils/parse_utils.py`)
2. Extract hyper-relational triples using `gemini-3-flash` (configurable in `kg_pipeline.py`)
3. Cluster entities via embedding similarity (SentenceTransformers + agglomerative clustering)
4. Adjudicate ambiguous clusters with an LLM for entity resolution
5. Output the final graph to `graphs/output_1000/`

Extraction results are cached per-chunk in `graphs/1000.jsonl`, so re-runs skip already-processed chunks.

**Output files:**
- `graphs/output_1000/graph.pkl` -- NetworkX graph (pickle)
- `graphs/output_1000/edges.jsonl` -- Edge list with hyper-relational properties
- `graphs/output_1000/nodes.jsonl` -- Node list with types
- `graphs/output_1000/canonical_map.json` -- Entity resolution mapping

### 3. Generate Questions (Optional)

To create a new QA benchmark from your corpus:

```bash
python generate_questions.py
```

This synthesizes questions across four reasoning categories (Lookup, Aggregation, Causal, Multi-hop) using a decoupled generation strategy. The included benchmark (`results/questions.jsonl`) contains the 252 expert-validated questions used in the paper.

### 4. Run the Evaluation

```bash
python run_experiment.py
```

This runs all model x method x k combinations defined in the configuration, writing results incrementally to `results/experiment_results.jsonl`. The experiment is resumable -- previously completed runs are skipped automatically.

**Configuration** (in `run_experiment.py`):
- `MODELS` -- LLM models to evaluate (default: Llama-3.1-8B/70B, GPT-OSS-20B/120B, Gemini-2.5-Pro)
- `K_SCHEDULE` -- Retrieval depths per method
- `MAX_WORKERS` -- Parallelism level (default: 8)

**Prerequisites for evaluation:**
- A built knowledge graph (Step 2) for Graph-RAG and Hybrid-RAG
- PDF source documents in `data/pdfs/` for Text-RAG
- Together AI API key for generation and judging

## Project Structure

```
.
├── kg_pipeline.py          # KG construction: extraction, clustering, resolution
├── engines.py              # Retrieval engines: Text-RAG, Graph-RAG, Hybrid-RAG
├── run_experiment.py       # Evaluation harness with dual-judge scoring
├── judge.py                # LLM-as-a-Judge with dual-judge consensus
├── llms.py                 # Provider-agnostic LLM interface + answer generation
├── generate_questions.py   # QA benchmark synthesis
├── config.py               # Shared configuration (paths, k-values, models)
├── prompts/
│   ├── extractor_prompt.txt      # Triple extraction prompt
│   ├── entity_resolution.txt     # Cluster adjudication prompt
│   └── refinement.txt            # Schema refinement prompt
├── utils/
│   ├── parse_utils.py      # PDF parsing, text chunking
│   ├── toc_utils.py         # Table of contents extraction
│   └── utils.py             # Text normalization, graph helpers
├── results/
│   └── questions.jsonl            # 252 expert-validated QA pairs
└── pyproject.toml
```

## Hyper-Relational Schema

Each edge in the knowledge graph carries seven qualifier types:

| Qualifier | Domain | Example |
|---|---|---|
| `condition` | Context | "If horse rushes" |
| `causality` | Mechanism | "To relax the jaw" |
| `instruction` | Technique | "Vibrate the hand" |
| `intensity` | Magnitude | Force/speed modifiers |
| `spatial_context` | Topology | "Behind the girth" |
| `frequency` | Rate | "Every 3 strides" |
| `modality` | Safety | Mandatory / Prohibited / Danger / Ideal / Mistake / Fact |

## Results Summary

Results from the paper (GPT-OSS-120B, accuracy-optimal k):

| Method | Accuracy | k* |
|---|---|---|
| Baseline (no retrieval) | 0.658 | -- |
| Graph-RAG | 0.681 | 50 |
| Hybrid-RAG | 0.862 | 5 |
| Text-RAG | 0.881 | 12 |
| Oracle (best per Q) | 0.913 | -- |

Graph-RAG is the uniquely best method for 9.5% of queries, primarily entity-centric lookups where competing values in the corpus cause text retrieval to surface the wrong answer. See Section 4.4 of the paper for detailed case studies.

## Citation

```bibtex
@inproceedings{davidsson2026graph,
  title={End-to-End Graph Retrieval Pipeline for Specialized Domains},
  author={Dav{\'\i}{\dh}sson, Haraldur Bjarni and Harmouch, Hazar},
  booktitle={Proceedings of the Workshop on Knowledge Graphs and Large Language Models (KG \& LLM 2026) @ LREC 2026},
  year={2026},
  address={Palma de Mallorca, Spain}
}
```

## License

This code is released for research purposes. The source corpus (Icelandic Riding Levels) is copyrighted material used with permission and is not included in this repository.
