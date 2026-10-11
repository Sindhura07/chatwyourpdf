from backend.citations import normalize_citations

VALID = {1, 2, 3, 4, 5}


def test_simple():
    out, used = normalize_citations("RAG is great [1].", VALID)
    assert out == "RAG is great [1]." and used == [1]


def test_grouped_formats():
    out, used = normalize_citations("A [Source ID: 1, Source ID: 2]. B [3, 4].", VALID)
    assert out == "A [1] [2]. B [3] [4]." and used == [1, 2, 3, 4]


def test_invalid_ids_dropped_and_non_citations_kept():
    out, used = normalize_citations("See [9] and [note].", VALID)
    assert "[note]" in out and used == []
