"""Split page text into overlapping word chunks, keeping source/page metadata."""


def chunk_pages(pages: list[dict], size: int = 180, overlap: int = 40, start_id: int = 0) -> list[dict]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks: list[dict] = []
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + size]
            if start > 0 and len(piece) <= overlap:
                break  # tiny trailing fragment already covered by the overlap
            chunks.append({
                "chunk_id": start_id + len(chunks),
                "source": page["source"],
                "page": page["page"],
                "text": " ".join(piece),
            })
    return chunks
