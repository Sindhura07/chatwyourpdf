import hashlib

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend import embeddings, llm


def fake_embed(texts, task="RETRIEVAL_DOCUMENT"):
    out = []
    for t in texts:
        v = np.zeros(64, dtype="float32")
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(out, dtype="float32")


def make_pdf(*page_texts) -> bytes:
    doc = pymupdf.open()
    for t in page_texts:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), t, fontsize=9)
    return doc.tobytes()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(llm, "generate", lambda q, chunks: "Answer [1, 2].")
    import backend.main as m
    m.SESSIONS.clear()
    return TestClient(m.app)


def test_upload_ask_and_isolation(client):
    pdf = make_pdf("alpha beta gamma retrieval generator " * 30, "dual encoder negatives training " * 30)
    a, b = {"X-Session": "session-aaaaaaaa"}, {"X-Session": "session-bbbbbbbb"}
    r = client.post("/upload", files={"file": ("p.pdf", pdf, "application/pdf")}, headers=a)
    assert r.status_code == 200 and r.json()["pages"] == 2
    assert [d["name"] for d in client.get("/documents", headers=a).json()["documents"]] == ["p.pdf"]
    assert client.get("/documents", headers=b).json()["documents"] == []  # other visitor sees nothing
    res = client.post("/ask", json={"question": "dual encoder negatives"}, headers=a).json()
    assert res["answer"] == "Answer [1] [2]." and res["sources"][0]["page"] == 2
    assert "Upload a PDF" in client.post("/ask", json={"question": "hi"}, headers=b).json()["answer"]
    client.delete("/documents/p.pdf", headers=a)
    assert client.get("/documents", headers=a).json()["documents"] == []


def test_rejects_bad_files(client):
    h = {"X-Session": "session-cccccccc"}
    assert client.post("/upload", files={"file": ("x.txt", b"hi", "text/plain")}, headers=h).status_code == 400
    assert client.post("/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}, headers=h).status_code == 400
    assert client.get("/documents").status_code == 400  # no session id
