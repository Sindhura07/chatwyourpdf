from backend.chunking import chunk_pages


def test_metadata_and_overlap():
    pages = [{"source": "a.pdf", "page": 3, "text": " ".join(f"w{i}" for i in range(500))}]
    chunks = chunk_pages(pages, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c["source"] == "a.pdf" and c["page"] == 3 for c in chunks)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0]["text"].split()[-20:] == chunks[1]["text"].split()[:20]


def test_start_id_offset():
    chunks = chunk_pages([{"source": "a.pdf", "page": 1, "text": "hello world"}], start_id=10)
    assert chunks[0]["chunk_id"] == 10
