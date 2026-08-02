"""FastAPI Main Entrypoint for AI-Powered Smart Retail & Customer Intelligence Platform.

Loads ML services and model artifacts once at startup into global application state
and mounts API routers for vision, nlp, and chatbot endpoints.
"""

import os
import sys
from contextlib import asynccontextmanager

# Add parent directory to sys.path to ensure module imports resolve across all environments
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import vision, nlp, chatbot
from app.services.cv_service import CVService
from app.services.nlp_service import NLPService
from app.services.chatbot_service import ChatbotService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize and load model pipelines ONCE at app startup."""
    print("==================================================")
    print(" [Startup] Initializing Smart Retail AI Platform Services...")
    print("==================================================")

    # Instantiate services once and store on app.state
    app.state.cv_service = CVService()
    app.state.nlp_service = NLPService()
    app.state.chatbot_service = ChatbotService()

    print(" [Startup] All service stubs & model pipelines loaded successfully.")
    print("==================================================")

    yield

    print(" [Shutdown] Cleaning up Smart Retail AI Platform resources...")


# Instantiate FastAPI app (Swagger docs automatically enabled at /docs)
app = FastAPI(
    title="AI-Powered Smart Retail & Customer Intelligence Platform",
    description=(
        "Enterprise retail intelligence API providing facial recognition customer check-in, "
        "MobileNetV2 product classification, NLTK-driven review sentiment analysis, "
        "and hybrid FAQ chatbot support."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Include API Routers
app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)

# Mount Static Files directory for Dashboard UI
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", tags=["Dashboard UI"])
async def root():
    """Serves the interactive Smart Retail AI Dashboard UI."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "platform": "Smart Retail & Customer Intelligence Platform",
        "documentation": "/docs"
    }


@app.get("/api/status", tags=["Health Check"])
async def api_status():
    """API health status endpoint."""
    return {
        "status": "online",
        "platform": "Smart Retail & Customer Intelligence Platform",
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
