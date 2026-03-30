import json
import string
import sys
from typing import Dict, List
import networkx as nx
import numpy as np
import time
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi

# Import your specific extractors/loaders from your existing files
from config import GRAPH_DATA_FILE
from utils.utils import normalize_text, extract_node_data


class BaseEngine:
    def retrieve(self, query: str, k: int) -> str:
        raise NotImplementedError


class BaselineEngine(BaseEngine):
    """No Retrieval - Just returns empty context"""

    def retrieve(self, query: str, k: int) -> str:
        return ""


class TextRAGEngine(BaseEngine):
    def __init__(
        self,
        passages,
        embedding_model_name="intfloat/multilingual-e5-base",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print("Initializing Optimal Text RAG (Hybrid + RRF + Reranking)...")
        self.passages = passages  # List of objects with 'text' attribute

        # 1. Bi-Encoder for Fast Retrieval
        self.bi_encoder = SentenceTransformer(embedding_model_name)

        # 2. Cross-Encoder for High-Precision Reranking
        # Note: Using the same reranker as GraphRAG ensures scientific fairness
        self.cross_encoder = CrossEncoder(reranker_model_name)

        # 3. Build Dense Index
        corpus = [p.text for p in self.passages]
        print(f"Encoding {len(corpus)} passages...")
        self.embeddings = self.bi_encoder.encode(
            corpus, convert_to_tensor=True, show_progress_bar=True
        )

        # 4. Build Sparse Index (BM25)
        print("Building BM25 Index...")
        tokenized_corpus = [normalize_text(doc).split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _reciprocal_rank_fusion(
        self, results_dict_list: List[Dict[int, float]], k=60
    ) -> Dict[int, float]:
        """
        Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
        score = 1 / (k + rank)
        """
        fused_scores = {}

        for results in results_dict_list:
            # Sort by score desc to determine rank
            sorted_res = sorted(results.items(), key=lambda item: item[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_res):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                fused_scores[doc_id] += 1.0 / (k + rank + 1)

        return fused_scores

    def retrieve(self, query: str, k: int) -> str:
        # We retrieve MORE than k initially (Candidate Generation) to allow Reranking to work
        candidate_k = max(k * 5, 50)

        # --- STEP 1: Dense Retrieval (Semantic) ---
        q_emb = self.bi_encoder.encode(query, convert_to_tensor=True)
        dense_res = util.semantic_search(q_emb, self.embeddings, top_k=candidate_k)[0]
        dense_hits = {hit["corpus_id"]: hit["score"] for hit in dense_res}

        # --- STEP 2: Sparse Retrieval (Keyword) ---
        tokenized_query = normalize_text(query).split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top candidate_k indices efficiently
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        sparse_hits = {idx: float(bm25_scores[idx]) for idx in top_bm25_indices}

        # --- STEP 3: Fusion (RRF) ---
        # Combine the two lists based on Rank, not raw score
        fused_scores = self._reciprocal_rank_fusion([dense_hits, sparse_hits])

        # Get top candidates for Reranking
        sorted_candidates = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )[:candidate_k]
        candidate_indices = [x[0] for x in sorted_candidates]

        # --- STEP 4: Reranking (Cross-Encoder) ---
        # Prepare pairs: [Query, Document Text]
        candidate_texts = [self.passages[idx].text for idx in candidate_indices]
        query_doc_pairs = [[query, doc] for doc in candidate_texts]

        # Predict scores (logits)
        rerank_scores = self.cross_encoder.predict(query_doc_pairs)

        # Sort by Re-ranker score
        final_results = sorted(
            zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True
        )

        # --- STEP 5: Selection ---
        top_k_indices = [idx for idx, score in final_results[:k]]
        retrieved_texts = [self.passages[i].text for i in top_k_indices]

        return "\n".join(retrieved_texts)


class GraphRAGEngine(BaseEngine):
    def __init__(self):
        print("Initializing Graph RAG...")
        # Make sure GRAPH_DATA_FILE is imported from your config

        self.G = nx.DiGraph()
        self.node_text_map = {}

        # --- FIX IS HERE: Capture the returned values ---
        self.G, self.node_text_map = self._load_graph(GRAPH_DATA_FILE)

        print(f"keys in node_text_map: {len(self.node_text_map)}")

        # Safety check to prevent ZeroDivisionError
        if not self.node_text_map:
            raise ValueError(
                f"Graph loaded from {GRAPH_DATA_FILE} is empty! Check your JSONL file."
            )

        # Models
        self.retriever = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Indices
        self.node_ids = list(self.node_text_map.keys())
        node_corpus = [self.node_text_map[nid] for nid in self.node_ids]

        print("Encoding Dense Vectors...")
        self.node_embeddings = self.retriever.encode(
            node_corpus, convert_to_tensor=True
        )

        print("Building BM25 Index...")
        tokenized = [doc.lower().split() for doc in node_corpus]
        self.bm25 = BM25Okapi(tokenized)

    def _load_graph(self, filepath):
        print(f"Loading graph from {filepath}...")
        G = nx.DiGraph()

        # Temporary storage to aggregate edge text per node
        node_context_buffer = {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        edge = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    src_raw = edge["source"]
                    tgt_raw = edge["target"]

                    # --- ROBUST NODE PARSING ---
                    src_id, src_label, src_name = extract_node_data(src_raw)
                    tgt_id, tgt_label, tgt_name = extract_node_data(tgt_raw)

                    # Add Nodes
                    G.add_node(src_id, label=src_label, name=src_name)
                    G.add_node(tgt_id, label=tgt_label, name=tgt_name)

                    # Extract Properties
                    props = edge.get("edge_properties", edge.get("props", {}))

                    # Add Edge to Graph
                    G.add_edge(src_id, tgt_id, **props, relation=edge["relation"])

                    # --- THESIS UPGRADE 1: Property Indexing ---
                    # Flatten properties into text string
                    prop_text = " ".join(
                        [
                            str(v)
                            for k, v in props.items()
                            if k not in ["source_reference"]
                        ]
                    )

                    # Initialize buffer if new
                    if src_id not in node_context_buffer:
                        node_context_buffer[src_id] = set()
                    if tgt_id not in node_context_buffer:
                        node_context_buffer[tgt_id] = set()

                    if prop_text:
                        # The Source is usually the subject, so it gets the context
                        node_context_buffer[src_id].add(prop_text)

        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
            sys.exit(1)

        # Build Final Search Dictionary {ID: Text}
        id_to_text_map = {}
        node_ids = list(G.nodes())

        for nid in node_ids:
            node = G.nodes[nid]
            # Handle cases where name might be missing
            name = node.get("name", str(nid))
            label = node.get("label", "Concept")

            base_text = f"{label}: {name}"
            # Add unique property context
            context_text = " ".join(node_context_buffer.get(nid, []))
            full_text = f"{base_text}. {context_text}".strip()
            id_to_text_map[nid] = full_text

        print(f"Graph Built: {len(G.nodes())} nodes. Context injected into embeddings.")
        return G, id_to_text_map

    def retrieve(self, query, k=10, top_k_seeds=7, alpha=0.85, debug=False):
        """
        Args:
            query: The search text.
            k: Number of final triples to return.
            top_k_seeds: How many entry points into the graph to find (Vector/Keyword).
            alpha: PageRank restart probability.
            debug: Print trace.

        Tuned Hyperparameters (via grid search, correctness=0.610):
            - sparse_weight: 0.4 (BM25 weight in fusion)
            - alpha: 0.85 (PageRank damping)
            - top_k_seeds: 7 (seed nodes for expansion)
            - subgraph_multiplier: 3 (limit = k * 3)
            - edge_rerank_limit: 70 (max edges to cross-encode)
        """
        if debug:
            print(f"\n{'='*40}\nDEBUGGING QUERY: {query}\n{'='*40}")

        # --- STEP 0: Global Keyword Safety Net ---
        direct_hits = []
        q_norm = normalize_text(query)
        translator = str.maketrans("", "", string.punctuation)
        q_clean = q_norm.translate(translator)

        search_terms = [
            t
            for t in q_clean.split()
            if len(t) > 3
            and t
            not in ["what", "how", "does", "fix", "caused", "affect", "best", "for"]
        ]

        # 1. Keyword Search
        for nid, text in self.node_text_map.items():
            t_norm = normalize_text(text)
            match_count = sum(1 for term in search_terms if term in t_norm)
            if match_count > 0:
                node_name = self.G.nodes[nid].get("name", "").lower()
                is_name_match = any(
                    term in normalize_text(node_name) for term in search_terms
                )
                score = match_count * 10.0 if is_name_match else match_count * 1.0
                direct_hits.append((nid, score))

        direct_hits.sort(key=lambda x: x[1], reverse=True)
        # Keep top seeds, but ensure we don't pick too many if they are weak
        seed_candidates = [x[0] for x in direct_hits[:15]]

        # --- STEP A: Hybrid Retrieval ---

        # 2. Dense Search
        query_emb = self.retriever.encode(query, convert_to_tensor=True)
        dense_hits = util.semantic_search(query_emb, self.node_embeddings, top_k=30)[0]
        dense_scores = {
            self.node_ids[hit["corpus_id"]]: hit["score"] for hit in dense_hits
        }

        # 3. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores_list = self.bm25.get_scores(tokenized_query)
        if max(bm25_scores_list) > 0:
            bm25_scores_list = bm25_scores_list / max(bm25_scores_list)
        bm25_scores = {
            self.node_ids[i]: score
            for i, score in enumerate(bm25_scores_list)
            if score > 0
        }

        # 4. Fusion (sparse_weight=0.4, dense_weight=0.6 — tuned)
        hybrid_scores = {}
        all_candidates = set(dense_scores.keys()) | set(bm25_scores.keys())

        for nid in all_candidates:
            d = dense_scores.get(nid, 0.0)
            s = bm25_scores.get(nid, 0.0)
            hybrid_scores[nid] = (0.6 * d) + (0.4 * s)  # Tuned: dense=0.6, sparse=0.4

        sorted_candidates = sorted(
            hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:20]

        # 5. Reranking (Cross-Encoder)
        candidate_ids_for_rerank = [x[0] for x in sorted_candidates]
        for hid in seed_candidates:
            if hid not in candidate_ids_for_rerank:
                candidate_ids_for_rerank.append(hid)

        candidate_full_texts = [
            self.node_text_map[nid] for nid in candidate_ids_for_rerank
        ]
        rerank_results = self.reranker.predict(
            [[query, text] for text in candidate_full_texts]
        )

        # Sigmoid normalization
        norm_scores = 1 / (1 + np.exp(-rerank_results))

        ranked_candidates = sorted(
            zip(candidate_ids_for_rerank, norm_scores), key=lambda x: x[1], reverse=True
        )

        # 6. Seed Selection Logic
        strong_keyword_seeds = [nid for nid, score in direct_hits if score >= 10.0]
        verified_seeds = []

        for nid, score in ranked_candidates:
            if nid in strong_keyword_seeds:
                if nid not in verified_seeds:
                    verified_seeds.append(nid)
            elif score > 0.05:  # Semantic threshold
                if nid not in verified_seeds:
                    verified_seeds.append(nid)

        if not verified_seeds:
            verified_seeds = [x[0] for x in ranked_candidates[:3]]

        final_seeds = verified_seeds[:top_k_seeds]

        # --- STEP B: Graph Expansion & Scoring ---

        # Dynamic Subgraph Limit (subgraph_multiplier=3 — tuned)
        limit_subgraph = max(50, k * 3) 

        personalization = {node: 0.0 for node in self.G.nodes()}
        
        for seed in final_seeds:
            if seed in personalization:
                personalization[seed] = 1.0

        try:
            ppr_scores = nx.pagerank(
                self.G, alpha=alpha, personalization=personalization
            )
        except ZeroDivisionError:
            ppr_scores = {n: 0 for n in self.G.nodes()}

        # Hub Penalty (Keep your existing logic, it is good)
        final_node_scores = {}
        for node, score in ppr_scores.items():
            degree = self.G.degree[node]
            # Slight tweak: prevent division by zero or negative
            penalty = np.log(degree + 2) 
            final_node_scores[node] = score / penalty

        # Select Top Nodes
        top_nodes = sorted(final_node_scores, key=final_node_scores.get, reverse=True)[:limit_subgraph]
        
        # Ensure seeds are included
        top_nodes = list(set(top_nodes + final_seeds))
        subgraph = self.G.subgraph(top_nodes)

        # --- STEP C: Edge Scoring & Reranking (MAJOR UPGRADE) ---

        candidate_edges = []
        
        for u, v, data in subgraph.edges(data=True):
            s_name = self.G.nodes[u].get("name", str(u))
            t_name = self.G.nodes[v].get("name", str(v))
            rel = data["relation"]
            
            # 1. Extract Properties (handle nested "properties" key)
            if "properties" in data and isinstance(data["properties"], dict):
                props = data["properties"]
            else:
                # Fallback: top-level props (excluding relation/source_reference)
                props = {k: v for k, v in data.items() if k not in ["relation", "source_reference", "properties"]}

            # Make the triple readable: "Rider --[CONTROLS]--> Speed (condition: In Tölt)"
            prop_text = f" ({json.dumps(props)})" if props else ""
            triple_str = f"{s_name} --[{rel}]--> {t_name}{prop_text}"

            # 2. Readable Sentence for Cross-Encoder (Better Semantic Match)
            # "Rider CONTROLS Speed. Context: condition is In Tölt"
            prop_sentence = ", ".join([f"{k} is {v}" for k, v in props.items()])
            readable_sentence = f"{s_name} {rel} {t_name}. {prop_sentence}"
            
            # 3. Base Heuristic Score (Keep your existing boost logic as a pre-filter)
            heuristic_score = 1.0
            if u in final_seeds or v in final_seeds:
                heuristic_score += 1.0
            
            # Save all data needed for reranking
            candidate_edges.append({
                "triple": triple_str,
                "readable": readable_sentence,
                "heuristic_score": heuristic_score
            })

        # Pre-filter before reranking (edge_rerank_limit=70 — tuned)
        candidate_edges.sort(key=lambda x: x["heuristic_score"], reverse=True)
        top_candidates = candidate_edges[:70] 
        # print(top_candidates) <--- remove or comment out

        # --- THE FIX STARTS HERE ---
        if not top_candidates:
            if debug:
                print("Graph traversal yielded no edges. Switching to Fallback Mode (Vector Node Search).")
            
            # Fallback: Just return the text of the top semantic matches (Vector RAG)
            # We use the 'final_seeds' which were the best matches from Step A
            fallback_nodes = final_seeds[:k] 
            fallback_triples = []
            
            for nid in fallback_nodes:
                # Retrieve the full text description we built during loading
                node_text = self.node_text_map.get(nid, "")
                # Format it to look like a "fact" so the LLM processes it similarly
                fallback_triples.append(f"Node Info: {node_text}")
            
            return self.verbalize(fallback_triples)
        # --- THE FIX ENDS HERE ---

        # FIX 3: Cross-Encoder Reranking of Edges
        # This fixes "Horse vs Rider" attribution errors.
        pairs_to_rank = [[query, item["readable"]] for item in top_candidates]
        
        # Use the same reranker you initialized in __init__
        rerank_scores = self.reranker.predict(pairs_to_rank)
        
        # Attach new scores
        for i, item in enumerate(top_candidates):
            item["semantic_score"] = rerank_scores[i]

        # Final Sort by Semantic Score
        top_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)

        # Slice Final K
        final_triples = [item["triple"] for item in top_candidates[:k]]
        return self.verbalize(final_triples)

    # Qualifier legend for LLM context
    QUALIFIER_LEGEND = """=== KNOWLEDGE GRAPH CONTEXT ===
        Each triple is formatted as: Source --[RELATION]--> Target (qualifiers)

        QUALIFIER KEY:
        - condition: When does this apply? (Context)
        - causality: Why? The mechanism or reason (Mechanism)
        - instruction: How exactly? Actionable technique (Technique)
        - intensity: Force/speed modifiers (Magnitude)
        - spatial_context: Location/anatomy (Topology)
        - frequency: Repetition pattern (Rate)
        - provenance: Source text reference (Grounding)
        - modality: Safety status - one of {Mandatory, Prohibited, Danger, Ideal, Mistake, Fact}

        TRIPLES:
        """

    def verbalize(self, triples_or_lines):
        """
        Converts graph output into text context for the LLM.
        Includes qualifier legend for better interpretation.
        """
        body = "\n".join([f"- {line}" for line in triples_or_lines])
        return self.QUALIFIER_LEGEND + body


class HybridRAGEngine(BaseEngine):
    def __init__(self, text_engine: TextRAGEngine, graph_engine: GraphRAGEngine):
        print("Initializing Hybrid RAG (Text + Graph)...")
        self.text_engine = text_engine
        self.graph_engine = graph_engine

    def retrieve(self, query: str, k: int) -> str:
        """
        Dynamically balances Text and Graph retrieval based on input k.

        Logic:
        - The input 'k' determines the number of Text Chunks (high token density).
        - We multiply 'k' by 10 to get the number of Graph Triples (low token density).

        Examples:
        - k=2  -> 2 Text Chunks  + 20 Graph Triples
        - k=5  -> 5 Text Chunks  + 50 Graph Triples
        - k=15 -> 15 Text Chunks + 150 Graph Triples
        """

        text_k = k
        graph_k = k * 10

        # 1. Retrieve Text (with error safety)
        try:
            text_context = self.text_engine.retrieve(query, k=text_k)
        except Exception as e:
            print(f"   [Hybrid Warning] Text retrieval failed: {e}")
            text_context = ""

        # 2. Retrieve Graph (with error safety)
        try:
            graph_context = self.graph_engine.retrieve(query, k=graph_k)
        except Exception as e:
            print(f"   [Hybrid Warning] Graph retrieval failed: {e}")
            graph_context = ""

        # 3. Combine with explicit headers for the LLM
        combined = (
            f"=== UNSTRUCTURED TEXT (Descriptive Context) ===\n{text_context}\n\n"
            f"=== STRUCTURED FACTS (Knowledge Graph) ===\n{graph_context}"
        )
        return combined
