"""
gateway/gateway.py
FastAPI entry point for the CampusFlow API.

Routes:
  POST /v1/chat           — main chat endpoint (session-aware)
  GET  /v1/session/{id}   — retrieve session state
  DELETE /v1/session/{id} — clear a session
  GET  /v1/universities   — list configured universities
  GET  /health            — health check
"""
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from config.loader import load_university_config, list_universities
from models.context import TenantContext
from models.session import Session
from orchestrator.dispatcher import chat_turn
from services.session_store import get_session_store

app = FastAPI(title="CampusFlow API", version="4.0.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""          # omit to start a new session
    user_id: str = ""


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    routing: dict
    responses: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    x_university_id: str = Header(..., description="University ID, e.g. 'utm'"),
):
    cfg = load_university_config(x_university_id)
    if not cfg:
        raise HTTPException(404, f"Unknown university: {x_university_id}")

    ctx = TenantContext(university_id=x_university_id, config=cfg)
    store = get_session_store()

    # Resolve or create session
    session_id = body.session_id or str(uuid.uuid4())
    session = store.get_or_create(session_id)

    result = chat_turn(body.message, session, ctx)
    store.save(session)

    return ChatResponse(
        session_id=session_id,
        reply=result.get("reply", ""),
        routing=result.get("routing", {}),
        responses=result.get("responses", {}),
    )


@app.get("/v1/session/{session_id}")
def get_session(session_id: str):
    store = get_session_store()
    session = store.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return session.to_dict()


@app.delete("/v1/session/{session_id}")
def delete_session(session_id: str):
    store = get_session_store()
    store.delete(session_id)
    return {"deleted": session_id}


@app.get("/v1/universities")
def get_universities():
    result = []
    for uid in list_universities():
        cfg = load_university_config(uid)
        if cfg:
            result.append({
                "id":          cfg.get("university_id"),
                "name":        cfg.get("display_name"),
                "departments": cfg.get("departments", []),
            })
    return result
