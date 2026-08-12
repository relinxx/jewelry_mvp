import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


load_dotenv()


class DatabaseConfigurationError(RuntimeError):
    pass


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise DatabaseConfigurationError("DATABASE_URL is not configured.")

    return database_url


@contextmanager
def get_connection():
    with psycopg.connect(get_database_url()) as connection:
        register_vector(connection)
        yield connection
