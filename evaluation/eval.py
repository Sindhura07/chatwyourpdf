"""Simple retrieval check on ONE pdf you choose.
Usage:  python -m evaluation.eval path/to/paper.pdf
Edit evaluation/questions.json so the keywords match that PDF.
"""
import json
import sys
from pathlib import Path

from backend.rag import Workspace


def main(pdf_path: str, k: int = 5):
    ws = Workspace()
    ws.add_pdf(Path(pdf_path).name, Path(pdf_path).read_bytes())
    cases = json.loads((Path(__file__).parent / "questions.json").read_text())
    hits = 0
    for case in cases:
        text = " ".join(c["text"].lower() for c in ws.retrieve(case["question"], k))
        ok = all(kw.lower() in text for kw in case["expect_keywords"])
        hits += ok
        print(("PASS" if ok else "FAIL"), "-", case["question"])
    print(f"\nRetrieval hit rate @{k}: {hits}/{len(cases)} = {hits / len(cases):.0%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m evaluation.eval path/to/file.pdf")
    main(sys.argv[1])
