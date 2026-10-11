# Chat with your PDF

Upload any PDF in the browser, ask questions, get answers with clickable page-level citations.
Each visitor gets a private workspace; nothing is saved to disk.

```
PDF upload → PyMuPDF text → chunks (+page) → Gemini embeddings → numpy search → Gemini answer
```

## Run locally (Windows PowerShell)
```powershell
cd ai-pro
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# .env must contain:  GEMINI_API_KEY=your-key
uvicorn backend.main:app
```
Open http://127.0.0.1:8000 and drop a PDF onto the page.

## Tests
```powershell
pip install pytest httpx
pytest -q
```

## API
| Method | Path | Notes |
|---|---|---|
| POST | `/upload` | multipart PDF, header `X-Session: <id>` |
| GET | `/documents` | list this visitor's PDFs |
| DELETE | `/documents/{name}` | remove one PDF |
| POST | `/ask` | `{"question": "..."}` |
| GET | `/health` | liveness check |

Limits (env vars): `MAX_UPLOAD_MB=25`, `MAX_CHUNKS=1500`, `MAX_SESSIONS=100`.
