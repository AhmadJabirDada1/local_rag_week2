from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv

jk
# The root folder of the Week 2 project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load variables from local_rag_week2/.env
load_dotenv(PROJECT_ROOT / ".env")


def get_int(name: str, default: int) -> int:
    """Read an integer environment variable safely."""
    value = os.getenv(name, str(default))

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer. Received: {value!r}"
        ) from exc


def get_float(name: str, default: float) -> float:
    """Read a floating-point environment variable safely."""
    value = os.getenv(name, str(default))

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number. Received: {value!r}"
        ) from exc


# Document settings
PDF_PATH = PROJECT_ROOT / os.getenv(
    "PDF_PATH",
    "data/rag-test-document.pdf",
)

CHUNK_SIZE = get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 120)

# Ollama settings
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "embeddinggemma",
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "llama3.2:3b",
)

EMBEDDING_BATCH_SIZE = get_int(
    "EMBEDDING_BATCH_SIZE",
    8,
)

# PostgreSQL settings
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = get_int("POSTGRES_PORT", 5432)
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
VECTOR_TABLE = os.getenv(
    "VECTOR_TABLE",
    "python_rag_chunks",
)

# Retrieval settings
TOP_K = get_int("TOP_K", 4)
MIN_SIMILARITY = get_float("MIN_SIMILARITY", 0.0)

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIRECTORY = PROJECT_ROOT / "logs"


def validate_settings() -> None:
    """Validate configuration before running the project."""

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    if EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError(
            "EMBEDDING_BATCH_SIZE must be greater than zero."
        )

    if TOP_K <= 0:
        raise ValueError("TOP_K must be greater than zero.")

    if not -1.0 <= MIN_SIMILARITY <= 1.0:
        raise ValueError(
            "MIN_SIMILARITY must be between -1 and 1."
        )

    # Table names cannot be passed as normal SQL parameters,
    # so restrict the value to a safe PostgreSQL identifier.
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*",
        VECTOR_TABLE,
    ):
        raise ValueError(
            "VECTOR_TABLE may contain only letters, numbers "
            "and underscores and cannot begin with a number."
        )


def setup_logging() -> None:
    """Configure console and file logging."""

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers if imported more than once.
    if root_logger.handlers:
        return

    numeric_level = getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    )

    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_DIRECTORY / "week2-rag.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


validate_settings()