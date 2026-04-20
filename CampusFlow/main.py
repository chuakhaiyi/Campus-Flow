"""
main.py — run from the CampusFlow_v4 root:

    python main.py              # interactive chat (default: utm)
    python main.py --test       # run the multi-turn test suite
    python main.py --uni ukm    # chat as a different university

API server:
    uvicorn gateway.gateway:app --reload

Mock university server (separate terminal):
    uvicorn mock_university_api.server:app --port 8001 --reload
"""
import sys
import os
import json
import asyncio
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import load_university_config
from models.context import TenantContext
from models.session import Session
from orchestrator.dispatcher import chat_turn

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_ctx(university_id: str) -> TenantContext:
    cfg = load_university_config(university_id)
    if not cfg:
        print(f"[ERROR] No config found for university: {university_id}")
        sys.exit(1)
    return TenantContext(university_id=university_id, config=cfg)


def _save_session(turns: list, session: Session) -> str:
    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(RESULTS_DIR, filename)
    payload = {
        "saved_at":     datetime.utcnow().isoformat() + "Z",
        "session_id":   session.session_id,
        "university_id": session.user_context.get("university_id", "utm"),
        "user_context": session.user_context,
        "tickets":      session.tickets,
        "turns":        turns,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# Suppress verbose internal debug prints in interactive / test mode
import io, contextlib

@contextlib.contextmanager
def _silent():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── Multi-turn test suite ─────────────────────────────────────────────────────

TEST_CONVERSATION = [
    "Bro I want to stay in the hostel this semester, are there any rooms available?",
    "I'm female and my budget is around RM250 per month",
    "What documents do I need to apply?",
    "Ok I want to apply for KM-B-103. My name is Siti Aisyah and my ID is STU-2024-0088",
]


def run_test(university_id: str = "utm") -> None:
    ctx = _build_ctx(university_id)
    session = Session()
    turns = []

    print("\n" + "=" * 70)
    print(f"  CampusFlow v4 — Multi-turn Test  [{ctx.config['display_name']}]")
    print("=" * 70)

    for msg in TEST_CONVERSATION:
        print(f"\n👤  {msg}")
        with _silent():
            result = chat_turn(msg, session, ctx)
        reply = result.get("reply", "")
        routing = result.get("routing", {})
        print(f"🤖  {reply}")
        print(f"    → {routing.get('departments', [])}  (confidence: {routing.get('confidence', 0):.2f}  followup: {routing.get('is_followup', False)})")
        turns.append({
            "user":      msg,
            "assistant": reply,
            "routing":   routing,
            "responses": result.get("responses", {}),
        })

    path = _save_session(turns, session)
    print(f"\n✅  Session saved → {path}")
    if session.tickets:
        print(f"🎫  Tickets raised: {session.tickets}")


# ── Interactive REPL ──────────────────────────────────────────────────────────

def run_chat(university_id: str = "utm") -> None:
    ctx = _build_ctx(university_id)
    session = Session()
    turns = []

    print("\n" + "=" * 70)
    print(f"  CampusFlow v4  |  {ctx.config['display_name']}")
    print("  Commands: 'exit' · 'new' (fresh session) · 'tickets' · 'context'")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            if turns:
                fp = _save_session(turns, session)
                print(f"✅  Session saved → {fp}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "exit":
            if turns:
                fp = _save_session(turns, session)
                print(f"\n✅  Session saved → {fp}")
            print("Goodbye!")
            break

        if cmd == "new":
            if turns:
                fp = _save_session(turns, session)
                print(f"✅  Session saved → {fp}\n")
            session = Session()
            turns = []
            print("[New session started]\n")
            continue

        if cmd == "tickets":
            if session.tickets:
                for dept, tid in session.tickets.items():
                    print(f"  {dept}: {tid}")
            else:
                print("  No tickets raised yet.")
            continue

        if cmd == "context":
            if session.user_context:
                for k, v in session.user_context.items():
                    print(f"  {k}: {v}")
            else:
                print("  No user context collected yet.")
            continue

        with _silent():
            result = chat_turn(user_input, session, ctx)

        reply = result.get("reply", "")
        routing = result.get("routing", {})
        print(f"\nCampusFlow: {reply}\n")
        print(f"  [→ {routing.get('departments', [])}  conf:{routing.get('confidence', 0):.2f}  followup:{routing.get('is_followup', False)}]\n")

        turns.append({
            "user":      user_input,
            "assistant": reply,
            "routing":   routing,
            "responses": result.get("responses", {}),
        })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusFlow v4")
    parser.add_argument("--test",  action="store_true", help="Run multi-turn test suite")
    parser.add_argument("--uni",   default="utm",       help="University ID (default: utm)")
    args = parser.parse_args()

    if args.test:
        run_test(args.uni)
    else:
        run_chat(args.uni)

print("Hello, CampusFlow v4 is ready to chat! Type your message below.\n")