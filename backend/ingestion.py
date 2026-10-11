"""PDF bytes -> list of pages with text. Nothing is written to disk."""
import pymupdf


def load_pdf_bytes(data: bytes, name: str) -> list[dict]:
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, start=1):
            text = " ".join(page.get_text().split())
            if text:
                pages.append({"source": name, "page": number, "text": text})
    return pages
