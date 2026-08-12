import os

import psycopg
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def main():

    if DATABASE_URL is None:
        raise ValueError("DATABASE_URL not found")

    with psycopg.connect(DATABASE_URL) as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS jewelry_items (
                    id BIGSERIAL PRIMARY KEY,

                    km_code VARCHAR(50)
                        UNIQUE
                        NOT NULL,

                    category VARCHAR(50)
                        NOT NULL,

                    image_path TEXT
                        NOT NULL,

                    embedding VECTOR(768)
                        NOT NULL
                );
                """
            )

        connection.commit()

    print("Database initialized successfully.")


if __name__ == "__main__":
    main()