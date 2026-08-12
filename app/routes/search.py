from io import BytesIO
import logging
import os

import psycopg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.database import DatabaseConfigurationError
from app.schemas import SearchQuery, SearchResponse
from app.services.similarity_search import search_similar_jewelry
from embedding_model import generate_embedding_from_image


logger = logging.getLogger(__name__)

router = APIRouter()

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_UPLOAD_BYTES = int(
    os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024))
)


def normalize_category(category):
    if category is None:
        return None

    category = category.strip().lower()

    if category in {"", "all"}:
        return None

    return category


async def read_upload(upload):
    size = 0
    chunks = []

    while True:
        chunk = await upload.read(1024 * 1024)

        if not chunk:
            break

        size += len(chunk)

        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded image is too large.",
            )

        chunks.append(chunk)

    return b"".join(chunks)


def open_supported_image(contents):
    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()

        image = Image.open(BytesIO(contents))
        image.load()

        return image
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        ) from error


@router.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    limit: int = Form(5, ge=1, le=20),
    category: str | None = Form(None),
):
    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supported image formats are JPEG, PNG, and WebP.",
        )

    contents = await read_upload(image)

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    pil_image = open_supported_image(contents)

    try:
        query_embedding = generate_embedding_from_image(pil_image)
    except Exception as error:
        logger.exception("Embedding generation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate an embedding for this image.",
        ) from error

    selected_category = normalize_category(category)

    try:
        results = search_similar_jewelry(
            query_embedding,
            limit=limit,
            category=selected_category,
        )
    except DatabaseConfigurationError as error:
        logger.exception("Database is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database is not configured.",
        ) from error
    except psycopg.Error as error:
        logger.exception("Similarity search failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Catalog search is unavailable.",
        ) from error

    return SearchResponse(
        query=SearchQuery(
            filename=image.filename or "uploaded-image",
            content_type=image.content_type or "unknown",
            limit=limit,
            category=selected_category,
        ),
        results=results,
    )
