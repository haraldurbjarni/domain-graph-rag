# config.py
import os

# Paths
DATA_DIR = "data"
RESULTS_DIR = "experiment_results"
PLOTS_DIR = "plots"
CACHE_DIR = "cache"
OUTPUT_FILE = "results/experiment_results.jsonl"
GRAPH_DATA_FILE = "graphs/output_1000/edges.jsonl"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Experiment Settings
K_VALUES = [5, 10, 20, 50, 100]
NUM_SAMPLES = 50  # 50 Simple, 50 Complex
K_SCHEDULE = {
    "text_rag": [1, 2, 3, 5, 7, 10, 12, 15],
    "graph_rag": [10, 15, 20, 30, 50, 70, 90],
    "hybrid_rag": [1, 2, 3, 5, 6],
    "baseline": [0],
}

# Model Definitions (Mapped to your Together/LLM providers)
MODELS = {
    "efficient_oss": "openai/gpt-oss-20b",  # Approx 8-20B class
    "large_oss": "openai/gpt-oss-120b",  # Represents larger reasoning models
    "proprietary": "gemini-2.5-pro",  # High-end
}
