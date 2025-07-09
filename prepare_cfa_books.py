# prepare_cfa_books.py

import os
from extractors import extract_text_from_pdf
from chunker import chunk_text
from embedding_db_cfa import store_chunks_cfa  # Use separate CFA DB

# Path to the CFA PDF file
CFA_PDF_PATH = "reference_docs/cfa_books/book_1.pdf"

def load_and_store_cfa_book():
    if not os.path.exists(CFA_PDF_PATH):
        print(f"❌ CFA book not found at: {CFA_PDF_PATH}")
        return

    print(f"📘 Loading CFA Book: {CFA_PDF_PATH}")
    with open(CFA_PDF_PATH, "rb") as f:
        text = extract_text_from_pdf(f)

    if not text.strip():
        print("❌ No text extracted from CFA book.")
        return

    print("✂️ Chunking CFA Book...")
    chunks = chunk_text(text)
    print(f"🧩 Total chunks: {len(chunks)}")

    print("💾 Storing chunks in CFA ChromaDB (namespace='cfa_books')...")
    stored, total = store_chunks_cfa(chunks, namespace="cfa_books")
    print(f"✅ Stored {stored} chunks. Total in CFA DB: {total}")

if __name__ == "__main__":
    load_and_store_cfa_book()
