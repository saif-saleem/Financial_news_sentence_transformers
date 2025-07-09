# embedding_db_cfa.py

import os
import uuid
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Store embeddings in a separate CFA directory
persist_dir = os.path.abspath("chroma_db_cfa")
chroma_client_cfa = PersistentClient(path=persist_dir)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_cfa_collection(namespace="cfa_books"):
    return chroma_client_cfa.get_or_create_collection(
        name=namespace,
        metadata={"hnsw:space": "cosine"}
    )

def store_chunks_cfa(chunks, metadatas=None, namespace="cfa_books"):
    if not chunks:
        print("⚠️ No CFA chunks to store.")
        return 0, get_cfa_collection(namespace).count()

    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embedder.encode(chunks).tolist()
    collection = get_cfa_collection(namespace)

    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    return len(chunks), collection.count()

def retrieve_cfa_chunks(query_text, top_k=10, namespace="cfa_books"):
    collection = get_cfa_collection(namespace)
    query_embedding = embedder.encode(query_text).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results.get("documents") else []
