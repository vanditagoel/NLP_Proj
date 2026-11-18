"""FastAPI application entrypoint serving REST APIs and generated Matplotlib PNG charts."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routes.api import router
from ml.utils import CHARTS_DIR, ensure_directories

ensure_directories()

app = FastAPI(
    title="Indic Climate Stance Detection API",
    description="CSET 346: Detecting Stance in Media on Global Warming for Indic Languages",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
