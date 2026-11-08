import json
import math
import os
import re
import sys
from pathlib import Path
import numpy as np

# Ensure parent and retrieval directories are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from sentence_transformers import SentenceTransformer

# Optional rank_bm25 with pure Python fallback
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


# ---------------------------------------------------------------------------
# Pure Python BM25 fallback (exact algorithmic match to BM25Okapi)
# ---------------------------------------------------------------------------
class PurePythonBM25:
    """Pure Python implementation of BM25Okapi matching rank_bm25 behavior."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0.0
        self.doc_freqs = []
        self.idf = {}

        # Count frequencies
        df = {}
        for doc in corpus:
            doc_counts = {}
            for token in doc:
                doc_counts[token] = doc_counts.get(token, 0) + 1
            self.doc_freqs.append(doc_counts)
            for token in doc_counts:
                df[token] = df.get(token, 0) + 1

        # Calculate IDF (Robertson-Spärck Jones)
        idf_sum = 0.0
        negative_idfs = []
        for word, freq in df.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        average_idf = idf_sum / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for word in query:
            if word not in self.idf:
                continue
            q_idf = self.idf[word]
            for idx in range(self.corpus_size):
                freq = self.doc_freqs[idx].get(word, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)) if self.avgdl > 0 else 1.0
                scores[idx] += q_idf * (num / den)
        return scores


# ---------------------------------------------------------------------------
# Utility: Tokenization
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Requirement 1: Load Embeddings Programmatically
# ---------------------------------------------------------------------------
def load_embeddings(file_path: str = None) -> list[dict]:
    """
    Safely locate and load embeddings.json into memory.
    Inspects candidate locations without dumping the raw content to stdout.
    """
    candidates = []
    if file_path:
        candidates.append(Path(file_path))

    candidates.extend([
        Path("retrieval/embeddings.json"),
        Path("embeddings.json"),
        Path("Rag-core/retrieval/embeddings.json"),
        Path("Rag-core/embeddings.json"),
        SCRIPT_DIR / "embeddings.json",
        SCRIPT_DIR.parent / "embeddings.json",
        SCRIPT_DIR.parent / "retrieval" / "embeddings.json",
    ])

    for path in candidates:
        if path.exists() and path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

    raise FileNotFoundError(
        f"Could not find embeddings file in any of: {[str(p) for p in candidates[:4]]}"
    )


# ---------------------------------------------------------------------------
# Requirement 2: Dense Retrieval (Cosine Similarity)
# ---------------------------------------------------------------------------
def cosine_similarity(vector1, vector2) -> float:
    """Calculates cosine similarity between two vectors."""
    v1 = np.asarray(vector1, dtype=np.float32)
    v2 = np.asarray(vector2, dtype=np.float32)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))


def dense_retrieval(query: str, chunks_data: list[dict], model=None, top_k: int = None) -> list[dict]:
    """
    Dense retrieval: compute query embedding and cosine similarity
    against pre-computed embeddings in chunks_data.
    """
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode(query, convert_to_numpy=True)

    ranked_results = []
    for chunk in chunks_data:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        ranked_results.append({
            "chunk_id": chunk["chunk_id"],
            "score": round(score, 6),
            "text": chunk.get("text", ""),
            "source_doc": chunk.get("source_doc", ""),
            "chunk_position": chunk.get("chunk_position", 0),
        })

    # Sort descending by cosine similarity score
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(ranked_results, start=1):
        item["dense_rank"] = rank

    if top_k is not None:
        return ranked_results[:top_k]
    return ranked_results


# ---------------------------------------------------------------------------
# Requirement 3: Sparse Retrieval (BM25)
# ---------------------------------------------------------------------------
def sparse_retrieval(query: str, chunks_data: list[dict], top_k: int = None) -> list[dict]:
    """
    Sparse retrieval: BM25 scoring over the document text corpus using
    rank_bm25 (if installed) or PurePythonBM25 fallback.
    """
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks_data]
    tokenized_query = tokenize(query)

    if BM25Okapi is not None:
        bm25_model = BM25Okapi(tokenized_corpus)
    else:
        bm25_model = PurePythonBM25(tokenized_corpus)

    scores = bm25_model.get_scores(tokenized_query)

    ranked_results = []
    for chunk, score in zip(chunks_data, scores):
        ranked_results.append({
            "chunk_id": chunk["chunk_id"],
            "score": round(float(score), 6),
            "text": chunk.get("text", ""),
            "source_doc": chunk.get("source_doc", ""),
            "chunk_position": chunk.get("chunk_position", 0),
        })

    # Sort descending by BM25 score
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(ranked_results, start=1):
        item["sparse_rank"] = rank

    if top_k is not None:
        return ranked_results[:top_k]
    return ranked_results


# ---------------------------------------------------------------------------
# Requirement 4: Reciprocal Rank Fusion (RRF)
# RRF_score(d) = sum(1 / (k + rank(d))) with default k = 60
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    dense_ranked: list[dict],
    sparse_ranked: list[dict],
    k: int = 60,
    top_k: int = None,
) -> list[dict]:
    """
    Combine dense and sparse rankings using Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum(1 / (k + rank(d)))
    
    where rank(d) is 1-based (rank 1 = highest ranked match).
    """
    rrf_scores = {}
    doc_lookup = {}
    dense_info = {}
    sparse_info = {}

    for rank, item in enumerate(dense_ranked, start=1):
        cid = item["chunk_id"]
        doc_lookup[cid] = item
        dense_info[cid] = {"dense_rank": rank, "dense_score": item["score"]}
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    for rank, item in enumerate(sparse_ranked, start=1):
        cid = item["chunk_id"]
        doc_lookup[cid] = item
        sparse_info[cid] = {"sparse_rank": rank, "sparse_score": item["score"]}
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_results = []
    for final_rank, cid in enumerate(sorted_doc_ids, start=1):
        item = doc_lookup[cid]
        d_meta = dense_info.get(cid, {"dense_rank": None, "dense_score": 0.0})
        s_meta = sparse_info.get(cid, {"sparse_rank": None, "sparse_score": 0.0})

        fused_results.append({
            "chunk_id": cid,
            "rrf_score": round(rrf_scores[cid], 6),
            "rank": final_rank,
            "source_doc": item.get("source_doc", ""),
            "dense_rank": d_meta["dense_rank"],
            "dense_score": d_meta["dense_score"],
            "sparse_rank": s_meta["sparse_rank"],
            "sparse_score": s_meta["sparse_score"],
            "text": item.get("text", ""),
        })

    if top_k is not None:
        return fused_results[:top_k]
    return fused_results


# ---------------------------------------------------------------------------
# High-Level Hybrid Retrieval Function
# ---------------------------------------------------------------------------
def hybrid_retrieve(
    query: str,
    chunks_data: list[dict],
    top_k: int = 5,
    k: int = 60,
    model=None,
) -> list[dict]:
    """Full hybrid retrieval pipeline: Dense + BM25 + RRF."""
    dense_ranked = dense_retrieval(query, chunks_data, model=model)
    sparse_ranked = sparse_retrieval(query, chunks_data)
    return reciprocal_rank_fusion(dense_ranked, sparse_ranked, k=k, top_k=top_k)


# ---------------------------------------------------------------------------
# Requirement 5: Test Execution & Output Demonstration
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  HYBRID RETRIEVAL PIPELINE (Dense + BM25 + Reciprocal Rank Fusion)  ")
    print("=" * 70)

    # 1. Programmatically load and inspect embeddings file
    chunks_data = load_embeddings()
    print(f"\n[1] Embeddings Loaded Successfully:")
    print(f"    - Total chunks loaded: {len(chunks_data)}")
    sample_chunk = chunks_data[0]
    print(f"    - Schema keys: {list(sample_chunk.keys())}")
    print(f"    - Embedding dimension: {len(sample_chunk.get('embedding', []))}")
    print(f"    - BM25 backend: {'rank_bm25' if BM25Okapi else 'PurePythonBM25 (fallback)'}")

    # 2. Sample query
    test_query = "What is the policy on flexible working hours?"
    print(f"\n[2] Sample Query: \"{test_query}\"")

    # Reusable embedding model instance
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Dense Retrieval
    dense_top = dense_retrieval(test_query, chunks_data, model=model, top_k=5)
    print("\n[3] Dense Retrieval (Cosine Similarity) - Top 5:")
    for r in dense_top:
        print(f"    Rank {r['dense_rank']}: {r['chunk_id']:<26} | Sim Score: {r['score']:.4f} | Source: {r['source_doc']}")

    # 4. Sparse Retrieval
    sparse_top = sparse_retrieval(test_query, chunks_data, top_k=5)
    print("\n[4] Sparse Retrieval (BM25) - Top 5:")
    for r in sparse_top:
        print(f"    Rank {r['sparse_rank']}: {r['chunk_id']:<26} | BM25 Score: {r['score']:.4f} | Source: {r['source_doc']}")

    # 5. Hybrid Retrieval (RRF with k=60)
    fused_results = hybrid_retrieve(test_query, chunks_data, top_k=5, k=60, model=model)

    print("\n" + "=" * 70)
    print("  FINAL TOP 5 RANKED DOCUMENTS (Reciprocal Rank Fusion, k=60)  ")
    print("=" * 70)
    for res in fused_results:
        print(f"\nRank {res['rank']} -> Chunk ID: {res['chunk_id']}")
        print(f"  RRF Score    : {res['rrf_score']:.6f}  [1/(60+{res['dense_rank']}) + 1/(60+{res['sparse_rank']})]")
        print(f"  Dense Rank   : #{res['dense_rank']} (Cosine Sim: {res['dense_score']:.4f})")
        print(f"  Sparse Rank  : #{res['sparse_rank']} (BM25 Score: {res['sparse_score']:.4f})")
        print(f"  Source Doc   : {res['source_doc']}")
        # Clean preview snippet
        clean_preview = res['text'].strip().replace("\n", " ")
        if len(clean_preview) > 140:
            clean_preview = clean_preview[:137] + "..."
        print(f"  Text Excerpt : \"{clean_preview}\"")

    print("\n" + "=" * 70)
    return fused_results


if __name__ == "__main__":
    main()

