from sentence_transformers import SentenceTransformer
from retrieval.similarity_search import find_similar_chunks


model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query: str, chunks_data: list, top_k: int = 5) -> list:
    
    query_embedding = model.encode(query).tolist()
    
   
    top_matches = find_similar_chunks(query_embedding, chunks_data, top_k=top_k)
    
    
    formatted_results = []
    for chunk, score in top_matches:
        formatted_results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source_doc": chunk["source_doc"],
            "similarity_score": round(float(score), 4)
        })
        
    return formatted_results