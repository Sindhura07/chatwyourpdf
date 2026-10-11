"""Text -> vectors using the Gemini embedding API (lightweight: no torch needed)."""
import time

import numpy as np

from backend import config

BATCH = 50
_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """task is RETRIEVAL_DOCUMENT for chunks and RETRIEVAL_QUERY for questions."""
    from google.genai import types

    client = _get_client()
    rows = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        for attempt in range(5):
            try:
                resp = client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task, output_dimensionality=config.EMBED_DIM
                    ),
                )
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (2 ** attempt))  # back off on rate limits
        rows.extend(item.values for item in resp.embeddings)
    arr = np.array(rows, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12  # normalize => dot product = cosine
    return arr
