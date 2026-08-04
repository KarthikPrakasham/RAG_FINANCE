"""
FastAPI application entry point.

Start the server (from the backend/ directory):
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or from the project root:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

API docs (auto-generated):
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import get_settings
from api.routers import chat, health, ingest, users

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Consumer Credit Rights Assistant API (model=%s)", settings.llm_model)
    # TODO: initialise DB connection pool, Pinecone client, BM25 index, etc.
    yield
    logger.info("Shutting down API")
    # TODO: clean up connections


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Consumer Credit Rights Assistant",
    description=(
        "RAG chatbot grounded in ECOA (Regulation B), the Fair Housing Act, "
        "and related Federal Reserve compliance chapters."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS — allow the Angular dev server and any configured origins
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Correlation-ID middleware
# Attaches X-Request-ID to every response for distributed tracing / logging.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next) -> Response:
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ---------------------------------------------------------------------------
# Global exception handler — never leak internal stack traces to the client
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = request.headers.get("X-Request-ID", "unknown")
    logger.exception("Unhandled error req_id=%s path=%s", req_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred.", "request_id": req_id},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, prefix="/api")           # GET /api/health, GET /api/metrics
app.include_router(chat.router, prefix="/api")             # POST /api/chat, GET /api/chat/history/{session_id}
app.include_router(ingest.router, prefix="/api")           # POST /api/ingest
app.include_router(users.router, prefix="/api")            # GET /api/users/{user_id}/profile
