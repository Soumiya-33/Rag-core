from fastembed import TextEmbedding
from retrieval.similarity_search import find_similar_chunks


class EmbeddingModel:
    def __init__(self):
        self.model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def encode(self, text):
        return list(self.model.embed([text]))[0]


model = EmbeddingModel()


def retrieve(query: str, chunks_data: list, top_k: int = 5) -> list:

    query_embedding = model.encode(query).tolist()

    top_matches = find_similar_chunks(
        query_embedding,
        chunks_data,
        top_k=top_k
    )

    formatted_results = []

    for chunk, score in top_matches:
        formatted_results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source_doc": chunk["source_doc"],
            "similarity_score": round(float(score), 4)
        })

    return formatted_results


def hybrid_retrieve(
    query: str,
    chunks_data: list,
    top_k: int = 5,
    k: int = 60
) -> list:
    """Convenience export for hybrid retrieval with Reciprocal Rank Fusion."""

    from retrieval.test_retrieval import hybrid_retrieve as _hybrid_retrieve

    return _hybrid_retrieve(
        query=query,
        chunks_data=chunks_data,
        top_k=top_k,
        k=k,
        model=model
    )
