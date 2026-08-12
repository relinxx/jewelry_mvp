import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routes.search import router as search_router


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
CATALOG_DIR = BASE_DIR / "output"


def get_cors_origins():
    frontend_url = os.getenv(
        "FRONTEND_URL",
        "http://localhost:5173",
    )

    return [
        origin.strip()
        for origin in frontend_url.split(",")
        if origin.strip()
    ]


app = FastAPI(
    title="Jewelry Similarity Search",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(
    search_router,
    prefix="/api",
)

app.mount(
    "/catalog",
    StaticFiles(directory=CATALOG_DIR, check_dir=False),
    name="catalog",
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR, check_dir=False),
    name="static",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Request validation failed.",
            "errors": exc.errors(),
        },
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
