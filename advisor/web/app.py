"""FastAPI web UI for tactical LLM advisor."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    query: str


# Set by __main__.py before starting
engine = None
listener = None
mesh_link = None


def create_app():
    app = FastAPI(title="Tactical Advisor")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / "index.html").read_text()

    @app.post("/api/query")
    async def query(req: QueryRequest):
        from advisor.context_builder import SYSTEM_PROMPT, build_context

        if engine is None:
            return JSONResponse(
                {"response": "LLM engine not initialised."},
                status_code=503,
            )

        state = listener.get_state() if listener else {}
        context = build_context(state)
        full_prompt = f"{context}\n\nQuery: {req.query}"

        # LLM calls block for seconds; keep them off the event loop
        response = await run_in_threadpool(engine.generate, SYSTEM_PROMPT, full_prompt)
        if not str(engine.last_backend or "").startswith("template"):
            from advisor.template_sitrep import ensure_categories

            response = ensure_categories(response, state)  # every nonzero category stays in
        return {"response": response, "backend": engine.last_backend}

    @app.get("/api/context")
    async def context():
        """The exact picture text the LLM is given (useful when demoing)."""
        from advisor.context_builder import build_context

        state = listener.get_state() if listener else {}
        return {"context": build_context(state)}

    @app.get("/api/status")
    async def status():
        state = listener.get_state() if listener else {}
        return {
            "model_loaded": engine.is_loaded if engine else False,
            "model_name": engine.model_name if engine else None,
            "tak_connected": listener.connected if listener else False,
            "positions": len(state.get("positions", {})),
            "messages": len(state.get("messages", [])),
            "alerts": len(state.get("alerts", [])),
            "drones": len(state.get("drones", [])),
            "sensor_events": len(state.get("sensor_events", [])),
            "sensor_assets": len(state.get("sensor_assets", [])),
            "llm": engine.status() if engine else None,
            "mesh": type(mesh_link).__name__ if mesh_link is not None else None,
            "mesh_connected": bool(getattr(mesh_link, "connected", False)),
        }

    return app