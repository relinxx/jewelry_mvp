import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from app.schemas import JewelrySearchResult

NPZ_FILE_PATH = Path("catalog_embeddings.npz")

# In-memory dataset cache
_cached_dataset = None


def get_dataset():
    global _cached_dataset
    if _cached_dataset is not None:
        return _cached_dataset

    if not NPZ_FILE_PATH.exists():
        return None

    data = np.load(NPZ_FILE_PATH)
    _cached_dataset = {
        "ids": data["ids"],
        "km_codes": data["km_codes"],
        "categories": data["categories"],
        "image_paths": data["image_paths"],
        "embeddings": data["embeddings"],  # Matrix shape (N, 768)
    }
    return _cached_dataset


def reload_dataset():
    global _cached_dataset
    _cached_dataset = None
    return get_dataset()


def catalog_image_url(image_path):
    path = Path(image_path)
    parts = path.parts

    if "output" not in parts:
        return f"/catalog/{quote(path.name)}"

    output_index = parts.index("output")
    relative_parts = parts[output_index + 1:]

    if not relative_parts:
        return f"/catalog/{quote(path.name)}"

    return "/" + "/".join(
        ["catalog", *[quote(part) for part in relative_parts]]
    )


def search_similar_jewelry(query_embedding, limit=5, category=None):
    dataset = get_dataset()

    if dataset is not None:
        embeddings = dataset["embeddings"]
        categories = dataset["categories"]
        ids = dataset["ids"]
        km_codes = dataset["km_codes"]
        image_paths = dataset["image_paths"]

        # Vectorized cosine similarity (dot product of L2 normalized vectors)
        similarities = embeddings @ query_embedding

        indices = np.arange(len(similarities))

        # Filter by category if specified
        if category:
            mask = np.char.lower(categories) == category.lower()
            indices = indices[mask]
            similarities = similarities[mask]

        if len(indices) == 0:
            return []

        # Sort descending by similarity score
        top_k_indices = np.argsort(-similarities)[:limit]

        results = []
        for idx in top_k_indices:
            orig_idx = indices[idx]
            path_str = str(image_paths[orig_idx])
            results.append(
                JewelrySearchResult(
                    id=int(ids[orig_idx]),
                    km_code=str(km_codes[orig_idx]),
                    category=str(categories[orig_idx]),
                    image_path=path_str,
                    image_url=catalog_image_url(path_str),
                    similarity=float(similarities[idx]),
                )
            )
        return results

    # Fallback to PostgreSQL if DATABASE_URL is set
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        from psycopg.rows import dict_row
        from app.database import get_connection

        with get_connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                if category:
                    cursor.execute(
                        """
                        SELECT
                            id,
                            km_code,
                            category,
                            image_path,
                            1 - (embedding <=> %s) AS similarity
                        FROM jewelry_items
                        WHERE category = %s
                        ORDER BY embedding <=> %s
                        LIMIT %s;
                        """,
                        (query_embedding, category, query_embedding, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT
                            id,
                            km_code,
                            category,
                            image_path,
                            1 - (embedding <=> %s) AS similarity
                        FROM jewelry_items
                        ORDER BY embedding <=> %s
                        LIMIT %s;
                        """,
                        (query_embedding, query_embedding, limit),
                    )

                return [
                    JewelrySearchResult(
                        id=row["id"],
                        km_code=row["km_code"],
                        category=row["category"],
                        image_path=row["image_path"],
                        image_url=catalog_image_url(row["image_path"]),
                        similarity=float(row["similarity"]),
                    )
                    for row in cursor.fetchall()
                ]

    raise RuntimeError(
        "No embedded catalog dataset found. Please run 'python embed_catalog.py' first to generate the dataset!"
    )

