import json
from retrieval.retrieve import retrieve
with open("embeddings.json", "r") as f:
    chunks_data = json.load(f)

test_query = "What is the policy on flexible working hours?"
results = retrieve(query=test_query, chunks_data=chunks_data, top_k=3)
print("--- Search Results ---")
print(json.dumps(results, indent=2))