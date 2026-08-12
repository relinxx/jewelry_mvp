from pathlib import Path
from urllib.parse import quote

from psycopg.rows import dict_row

from app.database import get_connection
from app.schemas import JewelrySearchResult


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


def row_to_result(row):
    return JewelrySearchResult(
        id=row["id"],
        km_code=row["km_code"],
        category=row["category"],
        image_path=row["image_path"],
        image_url=catalog_image_url(row["image_path"]),
        similarity=float(row["similarity"]),
    )


def search_similar_jewelry(query_embedding, limit=5, category=None):
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
                    (
                        query_embedding,
                        category,
                        query_embedding,
                        limit,
                    ),
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
                    (
                        query_embedding,
                        query_embedding,
                        limit,
                    ),
                )

            return [
                row_to_result(row)
                for row in cursor.fetchall()
            ]
