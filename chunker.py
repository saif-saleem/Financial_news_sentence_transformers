import re

def chunk_text(text, chunk_size=1000):
    # Split text into sentences using regex (handles '.', '?', '!' etc.)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
