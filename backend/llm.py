"""Step 6: Gemini answer generation."""
from backend import config

PROMPT = """You are a careful assistant that answers questions about the user's PDF. Answer the question using ONLY the numbered context below.
Rules:
- Cite sources inline like [1] or [2][3] right after the claim they support.
- If the context does not contain the answer, say you could not find it in the documents.
- Be clear and concise. Do not invent facts.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['rank']}] (file: {c['source']}, page {c['page']})\n{c['text']}" for c in chunks
    )
    return PROMPT.format(context=context, question=question)


def generate(question: str, chunks: list[dict]) -> str:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL, contents=build_prompt(question, chunks)
    )
    return (response.text or "").strip()
