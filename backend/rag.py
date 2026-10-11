"""Per-visitor RAG workspace: upload PDFs -> chunk -> embed -> search -> cited answer."""
import threading

from backend import config, embeddings, llm
from backend.chunking import chunk_pages
from backend.citations import normalize_citations
from backend.ingestion import load_pdf_bytes
from backend.vector_store import VectorStore


class Workspace:
    def __init__(self):
        self.store = VectorStore()
        self.docs: dict[str, int] = {}  # filename -> number of chunks
        self.lock = threading.Lock()

    def documents(self) -> list[dict]:
        return [{"name": n, "chunks": c} for n, c in self.docs.items()]

    def add_pdf(self, name: str, data: bytes) -> dict:
        pages = load_pdf_bytes(data, name)
        if not pages:
            raise ValueError("No readable text found. Scanned (image-only) PDFs are not supported.")
        with self.lock:
            self.remove(name)  # re-uploading the same file replaces it
            chunks = chunk_pages(pages, config.CHUNK_WORDS, config.CHUNK_OVERLAP, start_id=len(self.store.chunks))
            if len(self.store.chunks) + len(chunks) > config.MAX_CHUNKS:
                raise ValueError("This PDF is too large for the free demo. Try a smaller one or remove another file.")
            vectors = embeddings.embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
            self.store.add(chunks, vectors)
            self.docs[name] = len(chunks)
        return {"name": name, "pages": len(pages), "chunks": len(chunks)}

    def remove(self, name: str) -> None:
        if name in self.docs:
            self.store.remove_source(name)
            del self.docs[name]

    def retrieve(self, question: str, k: int = config.TOP_K) -> list[dict]:
        if self.store.vectors is None:
            return []
        qvec = embeddings.embed([question], "RETRIEVAL_QUERY")[0]
        return self.store.search(qvec, k)

    def ask(self, question: str, k: int = config.TOP_K) -> dict:
        chunks = self.retrieve(question, k)
        if not chunks:
            return {"answer": "Upload a PDF first, then ask me about it.", "sources": []}
        raw = llm.generate(question, chunks)
        answer, used = normalize_citations(raw, {c["rank"] for c in chunks})
        sources = [
            {"id": c["rank"], "file": c["source"], "page": c["page"],
             "score": round(c["score"], 3), "text": c["text"], "cited": c["rank"] in used}
            for c in chunks
        ]
        return {"answer": answer, "sources": sources}
