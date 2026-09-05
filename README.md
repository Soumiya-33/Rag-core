# Rag-core (learning based project)

RAG built from first principles — retrieval, embeddings, and evaluation, understood not just used.

## What is this?

A Retrieval-Augmented Generation (RAG) system that answers questions from a set of documents. Instead of just wiring together existing libraries, we build the core retrieval logic (chunking, embeddings, similarity search) from scratch first — then swap in production tools once we understand what they're actually doing.

## Why

Most beginner RAG projects import a vector DB and call it done. We wanted to actually learn the fundamentals — vectors, cosine similarity, retrieval evaluation — the concepts that show up across most real-world AI systems, not just this one.

## Stack

- **Embeddings:** sentence-transformers (local, free)
- **Retrieval:** numpy (from scratch) → ChromaDB/FAISS (v2)
- **Backend:** FastAPI
- **LLM:** Groq / Gemini (free tier)
- **Frontend:** React

## Status

🚧 v1 in progress — basic end-to-end pipeline (chunk → embed → retrieve → answer → display).

## Team

Built by four beginners, learning by doing.
