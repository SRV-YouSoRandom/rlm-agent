from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.logging import setup_logging
from core.tracing import setup_langsmith_tracing
from api.routes import ingest, chat, stream, collections, sessions, metrics
from services.vector_store import ensure_collection
import logging

setup_logging()
setup_langsmith_tracing()
logger = logging.getLogger("rlm_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RLM Agent v1.0.0...")
    ensure_collection()
    yield
    logger.info("Shutting down RLM Agent.")


app = FastAPI(
    title="RLM AI Chat Agent",
    description="Recursive Language Model Agent: REPL | Sub-LLM | Vector Search | Memory | Observability",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(stream.router, prefix="/api/v1", tags=["Streaming"])
app.include_router(collections.router, prefix="/api/v1", tags=["Collections"])
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "mode": "rlm"}