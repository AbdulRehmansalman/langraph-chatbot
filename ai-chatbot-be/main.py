"""
Production FastAPI Application
==============================
Enterprise-grade document chatbot API with health checks and graceful shutdown.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager

from app.api.routes import auth, documents, chat, google_auth
from app.health import routes as health
from app.core.middleware import LoggingMiddleware, RequestIDMiddleware
from app.core.config import settings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.

    Startup:
    - Log application start
    - Validate configuration
    - Initialize connections

    Shutdown:
    - Close database connections
    - Flush logs
    - Clean up resources
    """

    # ---- Startup ----
    logger.info(
        f"Starting Document Chatbot API v2.0.0 "
        f"in {settings.environment} mode"
    )
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Log level: {settings.log_level}")

    # Initialize resources here
    # Example:
    # await init_db()
    # await init_redis()

    yield  # Application runs here

    # ---- Shutdown ----
    logger.info("Shutting down Document Chatbot API...")

    # Clean up resources here
    # Example:
    # await close_db()
    # await close_redis()

    logger.info("Shutdown complete")


app = FastAPI(
    title="Document Chatbot API",
    description="Enterprise RAG system with advanced retrieval and streaming",
    version="2.0.0",
    lifespan=lifespan,
)

# --------------------
# Middleware
# --------------------

app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# --------------------
# Routes
# --------------------

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(google_auth.router, prefix="/api/auth", tags=["google-auth"])


@app.get("/")
async def root():
    return {
        "message": "Document Chatbot API is running!",
        "version": "2.0.0",
        "environment": settings.environment,
        "docs": "/docs",
    }


# --------------------
# Local development
# --------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Document Chatbot API server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )
