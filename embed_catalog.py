# embed_catalog.py

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from embedding_model import generate_embedding

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

CATALOG_DIRS = [
    Path("output/rings"),
    Path("output/bangles"),
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

OUTPUT_NPZ_PATH = Path("catalog_embeddings.npz")
OUTPUT_JSON_PATH = Path("catalog_metadata.json")


def get_km_code(image_path: Path) -> str:
    return image_path.stem


def get_category(image_path: Path) -> str:
    return image_path.parent.name


def find_catalog_images():
    images = []
    for catalog_dir in CATALOG_DIRS:
        if not catalog_dir.exists():
            continue
        for image_path in catalog_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            images.append(image_path)
    return images


def main():
    images = find_catalog_images()

    print(f"Found {len(images)} catalog images.")

    if not images:
        print("No catalog images found in output/ subdirectories.")
        return

    km_codes = []
    categories = []
    image_paths = []
    embeddings_list = []
    metadata = []

    for index, image_path in enumerate(images, start=1):
        km_code = get_km_code(image_path)
        category = get_category(image_path)

        print(f"[{index}/{len(images)}] Embedding {km_code} ({category})...")

        try:
            embedding = generate_embedding(image_path)  # shape (768,)
            km_codes.append(km_code)
            categories.append(category)
            image_paths.append(image_path.as_posix())
            embeddings_list.append(embedding)

            metadata.append({
                "id": index,
                "km_code": km_code,
                "category": category,
                "image_path": image_path.as_posix(),
            })
        except Exception as error:
            print(f"ERROR processing {image_path}: {error}")

    if not embeddings_list:
        print("No embeddings were generated.")
        return

    embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)

    # Save binary NumPy file for ultra-fast similarity calculations
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        ids=np.arange(1, len(km_codes) + 1, dtype=np.int64),
        km_codes=np.array(km_codes, dtype=str),
        categories=np.array(categories, dtype=str),
        image_paths=np.array(image_paths, dtype=str),
        embeddings=embeddings_matrix,
    )
    print(f"Saved binary dataset to {OUTPUT_NPZ_PATH.resolve()}")

    # Save JSON metadata
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata index to {OUTPUT_JSON_PATH.resolve()}")

    # Optional: If PostgreSQL DATABASE_URL is available, attempt to populate DB as well
    if DATABASE_URL:
        try:
            import psycopg
            from pgvector.psycopg import register_vector

            print("DATABASE_URL found, attempting to sync with PostgreSQL...")
            with psycopg.connect(DATABASE_URL) as connection:
                register_vector(connection)
                with connection.cursor() as cursor:
                    for km_code, category, path_str, emb in zip(
                        km_codes, categories, image_paths, embeddings_list
                    ):
                        cursor.execute(
                            """
                            INSERT INTO jewelry_items (km_code, category, image_path, embedding)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (km_code)
                            DO UPDATE SET
                                category = EXCLUDED.category,
                                image_path = EXCLUDED.image_path,
                                embedding = EXCLUDED.embedding;
                            """,
                            (km_code, category, path_str, emb),
                        )
                connection.commit()
            print("Successfully synced dataset to PostgreSQL!")
        except Exception as db_err:
            print(f"PostgreSQL sync skipped or failed (file dataset is ready): {db_err}")

    print("\nCatalog dataset preparation complete!")


if __name__ == "__main__":
    main()