import json
import time
import logging
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from llms import LLMJSON
from utils.parse_utils import load_passages

# Cache directory
CACHE_DIR = Path("./graphs")
CACHE_DIR.mkdir(exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import dotenv
dotenv.load_dotenv()

# ==========================================
# 1. CONFIGURATION & SCHEMA
# ==========================================

class PipelineConfig:
    """Controllable parameters for fine-tuning the pipeline."""
    def __init__(self):
        # Parallelism
        self.max_workers = 5  # Number of concurrent LLM calls
        
        # Consolidation Triggers
        self.consolidation_interval = 20  # Run deduplication every N chunks
        
        # Clustering Thresholds
        self.similarity_threshold = 0.5 # Distance threshold for Agglomerative Clustering (Lower = stricter)
        
        # Debugging
        self.mock_llm = True # Set False to use real API

class NodeInfo(BaseModel):
    name: str
    type: str
    abstraction_level: str = Field(description="Universal | Category | Instance")

class EdgeProperties(BaseModel):
    condition: Optional[str] = None
    causality: Optional[str] = None
    instruction: Optional[str] = None
    intensity: Optional[str] = None
    spatial_context: Optional[str] = None
    frequency: Optional[str] = None
    modality: str = Field(description="Mandatory | Danger | Ideal | Mistake | Fact")
    provenance: str

class HyperEdge(BaseModel):
    source: NodeInfo
    relation: str
    target: NodeInfo
    properties: EdgeProperties

class ExtractionResponse(BaseModel):
    triples: List[HyperEdge]

class ResolutionDecision(BaseModel):
    original_name: str
    action: str = Field(description="MERGE | KEEP | INSTANCE_OF")
    target_canonical: str

class ResolutionResponse(BaseModel):
    decisions: List[ResolutionDecision]



def load_prompt(name: str) -> str:
    """Utility to load prompt templates from files."""
    with open(f"prompts/{name}.txt", "r") as f:
        return f.read()


llmjson = LLMJSON(provider_hint="gemini", model_hint="gemini-3-flash-preview")
# ==========================================
# 2. THE LLM INTERFACE
# ==========================================

class LLMClient:
    """Wrapper for your specific LLM (e.g., Gemini-1.5-Flash)."""
    
    def extract_triples(self, chunk) -> ExtractionResponse:
        # Handle Passage objects or plain strings
        text = chunk.text if hasattr(chunk, 'text') else chunk

        user = load_prompt("extractor_prompt")
        sys_prompt = "You are an expert Knowledge Graph Engineer specializing in Icelandic Horse training and biomechanics. Your task is to extract a **Hyper-Relational Knowledge Graph** from the text provided."
        resp = llmjson(system=sys_prompt, prompt=user + text)
        print(f"LLM JSON Response: {resp}")

        # Wrap the raw list in ExtractionResponse
        if isinstance(resp, list):
            return ExtractionResponse(triples=[HyperEdge(**t) for t in resp])
        return resp

    def adjudicate_cluster(self, candidates: List[str]) -> ResolutionResponse:
        # PROMPT: The Judge
        system_prompt = "You are an expert Knowledge Graph Engineer specializing in entity resolution for Icelandic Horse training and biomechanics. Your task is to adjudicate whether the following candidate entity names refer to the same real-world entity or different ones, based on their context and semantics."
        user = load_prompt("entity_resolution") + str(candidates)
        resp = llmjson(system=system_prompt, prompt=user)

        # Parse response into ResolutionResponse
        if isinstance(resp, dict) and "decisions" in resp:
            decisions = []
            for d in resp["decisions"]:
                # For KEEP actions, target_canonical defaults to original_name
                target = d.get("target_canonical", d.get("original_name", ""))
                decisions.append(ResolutionDecision(
                    original_name=d.get("original_name", ""),
                    action=d.get("action", "KEEP"),
                    target_canonical=target
                ))
            return ResolutionResponse(decisions=decisions)
        elif isinstance(resp, list):
            # Handle case where LLM returns list directly
            decisions = []
            for d in resp:
                target = d.get("target_canonical", d.get("original_name", ""))
                decisions.append(ResolutionDecision(
                    original_name=d.get("original_name", ""),
                    action=d.get("action", "KEEP"),
                    target_canonical=target
                ))
            return ResolutionResponse(decisions=decisions)

        # Fallback: map all candidates to themselves
        logger.warning(f"Unexpected response format from LLM, keeping all as-is: {resp}")
        return ResolutionResponse(decisions=[
            ResolutionDecision(original_name=c, action="KEEP", target_canonical=c)
            for c in candidates
        ])

    def _mock_extraction(self, text):
        """Simulates LLM output for testing."""
        # Simple deterministic mock
        return ExtractionResponse(triples=[
            HyperEdge(
                source=NodeInfo(name="Rider", type="Entity", abstraction_level="Universal"),
                relation="CONTROLS",
                target=NodeInfo(name="Speed", type="Concept", abstraction_level="Universal"),
                properties=EdgeProperties(modality="Ideal", instruction="Use seat", provenance="p.10")
            ),
            HyperEdge(
                source=NodeInfo(name="The Rider", type="Entity", abstraction_level="Universal"), # Note synonym
                relation="CONTROLS",
                target=NodeInfo(name="Tempo", type="Concept", abstraction_level="Universal"),
                properties=EdgeProperties(modality="Mandatory", condition="In Tölt", provenance="p.12")
            )
        ])

    def _mock_adjudication(self, candidates):
        """Simulates LLM deciding that 'The Rider' == 'Rider'."""
        decisions = []
        canonical = min(candidates, key=len) # Heuristic: Shortest name is canonical
        for cand in candidates:
            if "Left" in cand or "Right" in cand: # Logic simulation: Keep sides distinct
                action = "KEEP"
                target = cand
            else:
                action = "MERGE"
                target = canonical
            
            decisions.append(ResolutionDecision(
                original_name=cand, 
                action=action, 
                target_canonical=target
            ))
        return ResolutionResponse(decisions=decisions)

# ==========================================
# 3. THE PIPELINE ENGINE
# ==========================================

class KnowledgeGraphPipeline:
    def __init__(self, config: PipelineConfig, chunk_size: int = 600):
        self.config = config
        self.chunk_size = chunk_size
        self.llm = LLMClient()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # State
        self.raw_triples_buffer: List[HyperEdge] = []
        self.canonical_map: Dict[str, str] = {} # "The Rider" -> "Rider"
        self.G = nx.MultiDiGraph() # The final artifact

        # Cache: chunk_hash -> list of triple dicts
        self.cache: Dict[str, List[Dict]] = {}
        self.cache_path = CACHE_DIR / f"{chunk_size}.jsonl"
        self._load_cache()

    def _load_cache(self):
        """Load extraction cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached extractions from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save extraction cache to disk."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.cache)} extractions to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    @staticmethod
    def _get_chunk_hash(chunk) -> str:
        """Generate a hash for a chunk based on its text content."""
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def run(self, text_chunks: List[str]):
        """Main execution flow."""
        logger.info(f"Starting pipeline with {len(text_chunks)} chunks")

        # 1. Parallel Extraction - process all chunks continuously
        self._extract_all(text_chunks)

        # 2. Print statistics before consolidation
        self._print_extraction_stats()

        # 3. Single consolidation pass at the end
        logger.info("Running consolidation...")
        self._consolidate_buffer()

        # 4. Final Polish
        self._finalize_graph()

        # 5. Save outputs to disk
        self._save_outputs()

        return self.G

    def _load_old_graph(self, path: str = "graph.jsonl") -> List[Dict]:
        """Load the old graph.jsonl for comparison."""
        old_triples = []
        old_path = Path(path)
        if not old_path.exists():
            return old_triples
        try:
            with open(old_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        old_triples.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load old graph: {e}")
        return old_triples

    def _get_old_graph_stats(self, old_triples: List[Dict]) -> Dict:
        """Extract statistics from old graph format."""
        nodes = set()
        relations = set()
        node_types = defaultdict(int)
        relation_counts = defaultdict(int)

        for t in old_triples:
            src_name = t.get("source", {}).get("name", "")
            tgt_name = t.get("target", {}).get("name", "")
            src_type = t.get("source", {}).get("label", "Unknown")
            tgt_type = t.get("target", {}).get("label", "Unknown")
            rel = t.get("relation", "")

            nodes.add(src_name)
            nodes.add(tgt_name)
            relations.add(rel)
            node_types[src_type] += 1
            node_types[tgt_type] += 1
            relation_counts[rel] += 1

        return {
            "total_triples": len(old_triples),
            "unique_nodes": len(nodes),
            "unique_relations": len(relations),
            "node_types": dict(node_types),
            "relation_counts": dict(relation_counts),
        }

    def _print_extraction_stats(self):
        """Print statistics about extracted triples before consolidation."""
        if not self.raw_triples_buffer:
            logger.info("No triples extracted.")
            return

        # Collect stats for new graph
        nodes = set()
        relations = set()
        node_types = defaultdict(int)
        relation_counts = defaultdict(int)
        modalities = defaultdict(int)

        for t in self.raw_triples_buffer:
            nodes.add(t.source.name)
            nodes.add(t.target.name)
            relations.add(t.relation)

            node_types[t.source.type] += 1
            node_types[t.target.type] += 1
            relation_counts[t.relation] += 1
            modalities[t.properties.modality] += 1

        new_stats = {
            "total_triples": len(self.raw_triples_buffer),
            "unique_nodes": len(nodes),
            "unique_relations": len(relations),
        }

        # Load old graph for comparison
        old_triples = self._load_old_graph()
        old_stats = self._get_old_graph_stats(old_triples) if old_triples else None

        # Print comparison
        print("\n" + "=" * 70)
        print("EXTRACTION STATISTICS (Before Consolidation)")
        print("=" * 70)

        if old_stats:
            print(f"{'Metric':<25} {'New Graph':>15} {'Old Graph':>15} {'Diff':>12}")
            print("-" * 70)
            for key, label in [("total_triples", "Total triples"),
                               ("unique_nodes", "Unique nodes"),
                               ("unique_relations", "Unique relations")]:
                new_val = new_stats[key]
                old_val = old_stats[key]
                diff = new_val - old_val
                diff_str = f"+{diff:,}" if diff >= 0 else f"{diff:,}"
                print(f"{label:<25} {new_val:>15,} {old_val:>15,} {diff_str:>12}")
        else:
            print(f"Total triples:      {new_stats['total_triples']:,}")
            print(f"Unique nodes:       {new_stats['unique_nodes']:,}")
            print(f"Unique relations:   {new_stats['unique_relations']:,}")
            print("\n(No old graph.jsonl found for comparison)")

        print(f"\nTop 10 Relations (New Graph):")
        for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {rel}: {count:,}")

        print(f"\nNode Types (New Graph):")
        for ntype, count in sorted(node_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {ntype}: {count:,}")

        print(f"\nModalities (New Graph):")
        for mod, count in sorted(modalities.items(), key=lambda x: -x[1]):
            print(f"  {mod}: {count:,}")

        print("=" * 70 + "\n")

    def _extract_all(self, chunks: List[str]):
        """Extract triples from all chunks with maximum parallelism, using cache."""
        # Separate cached vs uncached chunks
        uncached_chunks = []
        for chunk in chunks:
            chunk_hash = self._get_chunk_hash(chunk)
            if chunk_hash in self.cache:
                # Load from cache
                cached_triples = self.cache[chunk_hash]
                for t in cached_triples:
                    self.raw_triples_buffer.append(HyperEdge(**t))
            else:
                uncached_chunks.append((chunk_hash, chunk))

        logger.info(f"Cache hit: {len(chunks) - len(uncached_chunks)}, need to extract: {len(uncached_chunks)}")

        if not uncached_chunks:
            return

        # Extract uncached chunks in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._extract_and_cache, chunk_hash, chunk): chunk_hash
                for chunk_hash, chunk in uncached_chunks
            }

            for future in tqdm(as_completed(futures), total=len(uncached_chunks), desc="Extracting triples"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Extraction failed for chunk: {e}")

        # Save cache after extraction
        self._save_cache()

    def _extract_and_cache(self, chunk_hash: str, chunk):
        """Extract triples and store in cache."""
        result = self.llm.extract_triples(chunk)

        # Convert to dicts for JSON serialization
        triples_as_dicts = [t.model_dump() for t in result.triples]
        self.cache[chunk_hash] = triples_as_dicts

        # Also add to buffer
        self.raw_triples_buffer.extend(result.triples)

    def _consolidate_buffer(self):
        """
        Step 2 & 3: Vector Clustering + LLM Adjudication.
        This updates the self.canonical_map and pushes edges to the graph.
        """
        # A. Identify all new nodes in buffer
        node_names = set()
        for t in self.raw_triples_buffer:
            node_names.add(t.source.name)
            node_names.add(t.target.name)

        # Filter out nodes we've already resolved
        new_nodes = [n for n in node_names if n not in self.canonical_map]

        if not new_nodes:
            logger.info("No new nodes to consolidate.")
            return

        logger.info(f"Consolidating {len(new_nodes):,} unique nodes...")

        # B. Vector Clustering
        logger.info("Encoding node embeddings...")
        embeddings = self.embedder.encode(new_nodes, show_progress_bar=True)

        logger.info("Running agglomerative clustering...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.config.similarity_threshold,
            metric='cosine',
            linkage='average'
        )
        clustering.fit(embeddings)

        # Group by cluster ID
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            clusters[label].append(new_nodes[idx])

        # Categorize clusters by size
        single_node_clusters = [c for c in clusters.values() if len(c) == 1]
        pair_clusters = [c for c in clusters.values() if len(c) == 2]
        large_clusters = [c for c in clusters.values() if len(c) >= 3]

        logger.info(f"Found {len(clusters):,} clusters: "
                    f"{len(single_node_clusters):,} singletons, "
                    f"{len(pair_clusters):,} pairs (heuristic), "
                    f"{len(large_clusters):,} large (LLM)")

        # C1. Singletons - map to self
        for cluster_nodes in single_node_clusters:
            self.canonical_map[cluster_nodes[0]] = cluster_nodes[0]

        # C2. Pairs - use heuristic (shortest name is canonical)
        for cluster_nodes in tqdm(pair_clusters, desc="Resolving pairs (heuristic)", unit="pair"):
            canonical = min(cluster_nodes, key=len)
            for name in cluster_nodes:
                self.canonical_map[name] = canonical

        # C3. Large clusters (3+) - use LLM adjudication (parallelized)
        if large_clusters:
            def adjudicate_single_cluster(cluster_nodes):
                """Adjudicate a single cluster, return list of (original, canonical) pairs."""
                try:
                    resolution = self.llm.adjudicate_cluster(cluster_nodes)
                    return [(d.original_name, d.target_canonical) for d in resolution.decisions]
                except Exception as e:
                    logger.warning(f"LLM adjudication failed, using heuristic: {e}")
                    canonical = min(cluster_nodes, key=len)
                    return [(name, canonical) for name in cluster_nodes]

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(adjudicate_single_cluster, c): c for c in large_clusters}

                for future in tqdm(as_completed(futures), total=len(large_clusters),
                                   desc="LLM adjudication (3+ members)", unit="cluster"):
                    results = future.result()
                    for original, canonical in results:
                        self.canonical_map[original] = canonical

        logger.info(f"Consolidation complete. Canonical map has {len(self.canonical_map):,} entries.")

    def _finalize_graph(self):
        """
        Step 4: Hyper-Relational Logic Fusion.
        Merges the raw buffer into the NetworkX graph using the canonical map.
        """
        logger.info("Building Final Hyper-Relational Graph...")
        
        # Temporary storage for merging edges: (u, v, rel) -> {prop_key: set(values)}
        merged_edges = defaultdict(lambda: defaultdict(set))
        node_meta = {}

        for t in self.raw_triples_buffer:
            # 1. Resolve Names
            u = self.canonical_map.get(t.source.name, t.source.name)
            v = self.canonical_map.get(t.target.name, t.target.name)
            
            # 2. Store Node Metadata (Naive: Last write wins, or merging logic)
            if u not in node_meta: 
                node_meta[u] = {"type": t.source.type, "abstraction": t.source.abstraction_level}
            if v not in node_meta:
                node_meta[v] = {"type": t.target.type, "abstraction": t.target.abstraction_level}

            # 3. Merge Properties (The Hyper-Relational Magic)
            # We use a key of (u, v, relation) to group identical edges
            edge_key = (u, v, t.relation)
            
            # Iterate properties and add to sets (deduplication of values)
            props_dict = t.properties.model_dump(exclude_none=True)
            for k, val in props_dict.items():
                merged_edges[edge_key][k].add(val)

        # 4. Write to NetworkX
        # Add Nodes
        for n, meta in node_meta.items():
            self.G.add_node(n, **meta)
            
        # Add Edges
        for (u, v, rel), props_sets in merged_edges.items():
            # Convert sets to sorted lists for determinism
            final_props = {k: sorted(list(val_set)) for k, val_set in props_sets.items()}
            self.G.add_edge(u, v, relation=rel, **final_props)

        logger.info(f"Graph Built: {self.G.number_of_nodes()} Nodes, {self.G.number_of_edges()} Edges")

    def _save_outputs(self):
        """Save the final graph and canonical map to disk."""
        import pickle

        output_dir = CACHE_DIR / f"output_{self.chunk_size}"
        output_dir.mkdir(exist_ok=True)

        # 1. Save NetworkX graph (pickle - fast loading)
        graph_pkl_path = output_dir / "graph.pkl"
        with open(graph_pkl_path, 'wb') as f:
            pickle.dump(self.G, f)
        logger.info(f"Saved graph to {graph_pkl_path}")

        # 2. Save canonical map (JSON - inspectable)
        canonical_path = output_dir / "canonical_map.json"
        with open(canonical_path, 'w', encoding='utf-8') as f:
            json.dump(self.canonical_map, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved canonical map to {canonical_path}")

        # 3. Export edges as JSONL (for compatibility/inspection)
        edges_path = output_dir / "edges.jsonl"
        with open(edges_path, 'w', encoding='utf-8') as f:
            for u, v, data in self.G.edges(data=True):
                edge_data = {
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", ""),
                    "properties": {k: v for k, v in data.items() if k != "relation"}
                }
                f.write(json.dumps(edge_data, ensure_ascii=False) + "\n")
        logger.info(f"Saved {self.G.number_of_edges()} edges to {edges_path}")

        # 4. Export nodes as JSONL
        nodes_path = output_dir / "nodes.jsonl"
        with open(nodes_path, 'w', encoding='utf-8') as f:
            for node, data in self.G.nodes(data=True):
                node_data = {"name": node, **data}
                f.write(json.dumps(node_data, ensure_ascii=False) + "\n")
        logger.info(f"Saved {self.G.number_of_nodes()} nodes to {nodes_path}")

        logger.info(f"All outputs saved to {output_dir}/")

# ==========================================
# 4. RUNNER
# ==========================================

if __name__ == "__main__":
    # Load dataset
    chunk_size = 1000
    dataset = load_passages(chunk_size=chunk_size)

    # Configure parameters
    conf = PipelineConfig()
    conf.max_workers = 16

    # Run (pass chunk_size for cache path)
    pipeline = KnowledgeGraphPipeline(conf, chunk_size=chunk_size)
    final_graph = pipeline.run(dataset)
    
    # Inspect specific edge to prove Hyper-Relational merging
    # "Rider" and "The Rider" should be merged into "Rider"
    # "Speed" and "Tempo" might be merged (depending on mock logic)
    print("\n--- Edge Data Inspection ---")
    if final_graph.has_edge("Rider", "Speed"):
        print(final_graph.get_edge_data("Rider", "Speed"))