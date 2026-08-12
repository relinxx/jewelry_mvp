# embed_catalog.py

import os
from pathlib import Path

import psycopg

from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from embedding_model import generate_embedding


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

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


# -----------------------------------------------------
# Extract KM code from filename
# -----------------------------------------------------

def get_km_code(image_path):

    # Example:
    #
    # KM_409.jpg
    #
    # stem:
    #
    # KM_409

    return image_path.stem


# -----------------------------------------------------
# Figure out category from folder
# -----------------------------------------------------

def get_category(image_path):

    # Example:
    #
    # catalog/rings/KM_409.jpg
    #
    # parent.name:
    #
    # rings

    return image_path.parent.name


# -----------------------------------------------------
# Find catalog images
# -----------------------------------------------------

def find_catalog_images():

    images = []

    for catalog_dir in CATALOG_DIRS:

        for image_path in catalog_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            images.append(image_path)

    return images


# -----------------------------------------------------
# Main ingestion
# -----------------------------------------------------

def main():

    if DATABASE_URL is None:
        raise ValueError(
            "DATABASE_URL is missing."
        )

    images = find_catalog_images()

    print(
        f"Found {len(images)} "
        f"catalog images."
    )

    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        # Tell Psycopg about the vector type
        register_vector(
            connection
        )

        with connection.cursor() as cursor:

            for index, image_path in enumerate(
                images,
                start=1,
            ):

                km_code = get_km_code(
                    image_path
                )

                category = get_category(
                    image_path
                )

                print(
                    f"[{index}/{len(images)}] "
                    f"Embedding {km_code}"
                )

                try:

                    embedding = (
                        generate_embedding(
                            image_path
                        )
                    )

                    cursor.execute(
                        """
                        INSERT INTO jewelry_items (
                            km_code,
                            category,
                            image_path,
                            embedding
                        )

                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s
                        )

                        ON CONFLICT (km_code)

                        DO UPDATE SET
                            category =
                                EXCLUDED.category,

                            image_path =
                                EXCLUDED.image_path,

                            embedding =
                                EXCLUDED.embedding;
                        """,
                        (
                            km_code,
                            category,
                            str(image_path),
                            embedding,
                        )
                    )

                except Exception as error:

                    print(
                        f"ERROR processing "
                        f"{image_path}: "
                        f"{error}"
                    )

            connection.commit()

    print(
        "\nCatalog embedding complete."
    )


if __name__ == "__main__":
    main()