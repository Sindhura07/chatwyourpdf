"""FastAPI server: serves the UI and a private workspace per visitor (identified by X-Session)."""
import re
from collections import OrderedDict

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import config
from backend.rag import Workspace

app = FastAPI(title="Chat with your PDF", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: "OrderedDict[str, Workspace]" = OrderedDict()


def workspace(sid: str) -> Workspace:
    if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", sid or ""):
        raise HTTPException(status_code=400, detail="Missing or invalid session id.")
    ws = SESSIONS.get(sid)
    if ws is None:
        ws = SESSIONS[sid] = Workspace()
        while len(SESSIONS) > config.MAX_SESSIONS:
            SESSIONS.popitem(last=False)
    else:
        SESSIONS.move_to_end(sid)
    return ws


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents(x_session: str = Header(default="")):
    return {"documents": workspace(x_session).documents()}


@app.post("/upload")
def upload(file: UploadFile = File(...), x_session: str = Header(default="")):
    ws = workspace(x_session)
    name = (file.filename or "document.pdf").replace("/", "_").replace("\\", "_")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="That file doesn't look like a valid PDF.")
    try:
        return ws.add_pdf(name, data)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except RuntimeError as err:  # e.g. missing API key
        raise HTTPException(status_code=500, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=502, detail=f"Could not process this PDF: {str(err)[:150]}")


@app.delete("/documents/{name}")
def delete_document(name: str, x_session: str = Header(default="")):
    workspace(x_session).remove(name)
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest, x_session: str = Header(default="")):
    ws = workspace(x_session)
    try:
        return ws.ask(req.question, req.top_k)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=502, detail=f"The AI service had a problem: {str(err)[:150]}")
