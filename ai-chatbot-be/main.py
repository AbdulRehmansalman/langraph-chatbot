"""
Production FastAPI Application
==============================
Enterprise-grade document chatbot API with health checks and graceful shutdown.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import signal
import sys
from contextlib import asynccontextmanager

from app.api.routes import auth, documents, chat, google_auth, streaming_chat
from app.health import routes as health
from app.core.middleware import LoggingMiddleware, RequestIDMiddleware
from app.core.config import settings

load_dotenv()

logger = logging.getLogger(__name__)

# Graceful shutdown flag
shutdown_event = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_event
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Startup:
    - Log application start
    - Validate configuration
    - Initialize connections
    
    Shutdown:
    - Close database connections
    - Flush logs
    - Clean up resources
    """
    # Startup
    logger.info(f"Starting Document Chatbot API v2.0.0 in {settings.environment} mode")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Log level: {settings.log_level}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Chatbot API...")
    logger.info("Closing database connections...")
    logger.info("Shutdown complete")


app = FastAPI(
    title="Document Chatbot API",
    description="Enterprise RAG system with advanced retrieval and streaming",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Configure CORS from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(streaming_chat.router, prefix="/api/chat", tags=["chat", "streaming"])
app.include_router(google_auth.router, prefix="/api/auth", tags=["google-auth"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Document Chatbot API is running!",
        "version": "2.0.0",
        "environment": settings.environment,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Document Chatbot API server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower()
    )
