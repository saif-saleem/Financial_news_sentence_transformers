# embedding_db.py

import os
import uuid
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Initialize persistent Chroma client
persist_dir = os.path.abspath("chroma_db")
chroma_client = PersistentClient(path=persist_dir)

# ✅ SentenceTransformer embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Get or create collection by namespace
def get_chroma_collection(namespace="financial_news"):
    return chroma_client.get_or_create_collection(
        name=namespace,
        metadata={"hnsw:space": "cosine"}
    )

# Store text chunks in specified namespace
def store_chunks(chunks, metadatas=None, namespace="financial_news"):
    if not chunks:
        print("⚠️ No chunks to store.")
        return 0, get_chroma_collection(namespace).count()

    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embedder.encode(chunks).tolist()
    collection = get_chroma_collection(namespace)

    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    stored_count = len(chunks)
    total_count = collection.count()
    print(f"✅ Stored {stored_count} chunks in '{namespace}'. Total in DB: {total_count}")
    return stored_count, total_count

# Retrieve similar chunks using default financial_news namespace
def retrieve_similar_chunks(query_text, top_k=5):
    return retrieve_with_context(query_text, top_k=top_k, namespace="financial_news")

# Generic retrieval from any namespace
def retrieve_with_context(query_text, top_k=5, namespace="financial_news"):
    collection = get_chroma_collection(namespace)
    query_embedding = embedder.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0] if results.get("documents") else []

# Peek all docs in default collection
def get_all_documents():
    collection = get_chroma_collection("financial_news")
    results = collection.peek()
    count = len(results["ids"])
    return count, None
