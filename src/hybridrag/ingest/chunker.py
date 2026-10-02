from typing import List

def sliding_window_chunk(text: str, window_size: int = 600, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= window_size:
        return [text]
        
    chunks: List[str] = []
    step = window_size - overlap
    i = 0
    while i < len(text):
        chunk = text[i:i + window_size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
        if len(text) - i < step:
            final_chunk = text[i:].strip()
            if final_chunk and final_chunk not in chunks:
                chunks.append(final_chunk)
            break
    return chunks
