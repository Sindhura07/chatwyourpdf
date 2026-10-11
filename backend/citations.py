"""Step 5: robust citation handling.

The LLM is asked to cite as [1], [2]. Models sometimes write [1, 2], [Source 1],
or [Source ID: 1, Source ID: 2]. normalize_citations() turns all of these into
clean "[1] [2]" markers and returns which source numbers were actually used.
"""
import re

_BRACKET = re.compile(r"\[([^\[\]]*?\d[^\[\]]*?)\]")


def normalize_citations(answer: str, valid_ids: set[int]) -> tuple[str, list[int]]:
    used: list[int] = []

    def repl(match: re.Match) -> str:
        inner = match.group(1)
        # only touch brackets that look like citations
        if not re.fullmatch(r"(?:\s*(?:source\s*(?:id)?\s*:?\s*)?\d+\s*[,;]?)+", inner, re.I):
            return match.group(0)
        ids = [int(n) for n in re.findall(r"\d+", inner)]
        ids = [i for i in ids if i in valid_ids]
        for i in ids:
            if i not in used:
                used.append(i)
        return " ".join(f"[{i}]" for i in ids)

    return _BRACKET.sub(repl, answer), used
