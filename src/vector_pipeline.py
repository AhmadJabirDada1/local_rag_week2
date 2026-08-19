from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import psycopg
from ollama import Client, ResponseError
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    PDF_PATH,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    TOP_K,
    VECTOR_TABLE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    setup_logging,
)

from .document_pipeline import (
    TextChunk,
    load_and_chunk_pdf,
)


logger = logging.getLogger(__name__)


def create_ollama_client() -> Client:
    """
    Create a client connected to the local Ollama service.
    """

    return Client(host=OLLAMA_HOST)


def connect_to_postgres() -> psycopg.Connection:
    """
    Connect Python to the local PostgreSQL database,
    enable pgvector and register vector handling.
    """

    if not POSTGRES_PASSWORD:
        raise ValueError(
            "POSTGRES_PASSWORD is missing from .env."
        )

    try:
        connection = psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=5,
            autocommit=True,
            row_factory=dict_row,
        )

        connection.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )

        register_vector(connection)

        logger.info(
            "Connected to PostgreSQL database %s "
            "at %s:%d",
            POSTGRES_DB,
            POSTGRES_HOST,
            POSTGRES_PORT,
        )

        return connection

    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to PostgreSQL. Check that "
            "Docker is running, PostgreSQL is healthy and "
            "the .env credentials are correct."
        ) from exc


def generate_embeddings(
    client: Client,
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Convert one or more text strings into embeddings.
    """

    if not texts:
        return []

    try:
        response = client.embed(
            model=EMBEDDING_MODEL,
            input=list(texts),
        )

    except ResponseError as exc:
        raise RuntimeError(
            f"Ollama embedding request failed: "
            f"{exc.error}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Confirm that "
            "Ollama is running and that embeddinggemma "
            "is installed."
        ) from exc

    if hasattr(response, "embeddings"):
        raw_embeddings = response.embeddings
    else:
        raw_embeddings = response["embeddings"]

    embeddings = [
        [
            float(value)
            for value in embedding
        ]
        for embedding in raw_embeddings
    ]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} "
            f"embeddings for {len(texts)} text inputs."
        )

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    if len(dimensions) != 1:
        raise RuntimeError(
            "The generated embeddings do not have "
            "one consistent dimension."
        )

    return embeddings


def generate_chunk_embeddings(
    client: Client,
    chunks: Sequence[TextChunk],
) -> list[list[float]]:
    """
    Embed document chunks in small batches.
    """

    if not chunks:
        raise ValueError(
            "No chunks were supplied for embedding."
        )

    all_embeddings: list[list[float]] = []
    total_chunks = len(chunks)

    for batch_start in range(
        0,
        total_chunks,
        EMBEDDING_BATCH_SIZE,
    ):
        batch = chunks[
            batch_start:
            batch_start + EMBEDDING_BATCH_SIZE
        ]

        logger.info(
            "Embedding chunks %d to %d of %d",
            batch_start + 1,
            min(
                batch_start + len(batch),
                total_chunks,
            ),
            total_chunks,
        )

        batch_embeddings = generate_embeddings(
            client=client,
            texts=[
                chunk.text
                for chunk in batch
            ],
        )

        all_embeddings.extend(
            batch_embeddings
        )

    logger.info(
        "Generated %d embeddings.",
        len(all_embeddings),
    )

    logger.info(
        "Embedding vector dimension: %d",
        len(all_embeddings[0]),
    )

    return all_embeddings


def create_vector_table(
    connection: psycopg.Connection,
    dimension: int,
    reset: bool,
) -> None:
    """
    Create the Week 2 Python vector table.

    --reset drops the table first so that changed chunking
    or embedding dimensions cannot leave stale records.
    """

    if dimension <= 0:
        raise ValueError(
            "Embedding dimension must be greater than zero."
        )

    if reset:
        connection.execute(
            f"DROP TABLE IF EXISTS {VECTOR_TABLE}"
        )

        logger.warning(
            "Existing table %s was removed.",
            VECTOR_TABLE,
        )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_length INTEGER NOT NULL,
            embedding vector({dimension}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
                DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (
                source,
                page_number,
                chunk_index
            )
        )
        """
    )

    logger.info(
        "Vector table %s is ready.",
        VECTOR_TABLE,
    )


def store_chunks(
    connection: psycopg.Connection,
    chunks: Sequence[TextChunk],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """
    Store chunk text, source data and embedding vectors.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Received {len(chunks)} chunks but "
            f"{len(embeddings)} embeddings."
        )

    inserted_or_updated = 0

    insert_query = f"""
        INSERT INTO {VECTOR_TABLE} (
            source,
            page_number,
            chunk_index,
            content,
            content_length,
            embedding
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT (
            source,
            page_number,
            chunk_index
        )
        DO UPDATE SET
            content = EXCLUDED.content,
            content_length = EXCLUDED.content_length,
            embedding = EXCLUDED.embedding,
            created_at = CURRENT_TIMESTAMP
    """

    with connection.cursor() as cursor:
        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            clean_content = chunk.text.replace("\x00", "")
            cursor.execute(
                insert_query,
                (
                    chunk.source,
                    chunk.page_number,
                    chunk.chunk_index,
                    clean_content,
                    len(clean_content),
                    Vector(embedding),
                ),
            )

            inserted_or_updated += 1

    logger.info(
        "Stored or updated %d vector rows.",
        inserted_or_updated,
    )

    return inserted_or_updated


def count_vector_rows(
    connection: psycopg.Connection,
) -> int:
    """
    Count the rows currently stored in the vector table.
    """

    result = connection.execute(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {VECTOR_TABLE}
        """
    ).fetchone()

    return int(result["row_count"])


def index_document(reset: bool) -> None:
    """
    Execute the complete Thursday indexing process.
    """

    pdf_files = sorted(
    DATA_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}"
        )

    all_pages = []
    all_chunks = []

    print(
        f"\nFound {len(pdf_files)} PDF document(s)."
    )

    for pdf_file in pdf_files:

        print(
            f"\nProcessing: {pdf_file.name}"
        )

        pages, chunks = load_and_chunk_pdf(
            pdf_path=pdf_file,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        all_pages.extend(pages)
        all_chunks.extend(chunks)

        print(
            f"Pages containing text: {len(pages)}"
        )

        print(
            f"Chunks created: {len(chunks)}"
        )

    pages = all_pages
    chunks = all_chunks

    ollama_client = create_ollama_client()

    embeddings = generate_chunk_embeddings(
        client=ollama_client,
        chunks=chunks,
    )

    embedding_dimension = len(
        embeddings[0]
    )

    connection = connect_to_postgres()

    try:
        create_vector_table(
            connection=connection,
            dimension=embedding_dimension,
            reset=reset,
        )

        stored_count = store_chunks(
            connection=connection,
            chunks=chunks,
            embeddings=embeddings,
        )

        total_rows = count_vector_rows(
            connection
        )

    finally:
        connection.close()

    print("\nINDEXING SUMMARY")
    print("=" * 75)

    print(
        f"Documents processed: {len(pdf_files)}"
    )

    print(
        f"Pages containing text: {len(pages)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print(
        f"Embeddings generated: {len(embeddings)}"
    )

    print(
        f"Embedding dimension: "
        f"{embedding_dimension}"
    )

    print(
        f"Rows stored or updated: "
        f"{stored_count}"
    )

    print(
        f"Total rows in {VECTOR_TABLE}: "
        f"{total_rows}"
    )


def retrieve_chunks(
    connection: psycopg.Connection,
    client: Client,
    question: str,
    top_k: int,
) -> list[dict]:
    """
    Embed a question and retrieve its Top-K closest chunks.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "The question cannot be blank."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    question_embedding = generate_embeddings(
        client=client,
        texts=[question],
    )[0]

    vector = Vector(
        question_embedding
    )

    query = f"""
        SELECT
            id,
            source,
            page_number,
            chunk_index,
            content,
            content_length,
            1 - (embedding <=> %s) AS similarity
        FROM {VECTOR_TABLE}
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    rows = connection.execute(
        query,
        (
            vector,
            vector,
            top_k,
        ),
    ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def run_query(
    question: str,
    top_k: int,
) -> None:
    """
    Run retrieval without the final LLM.
    """

    client = create_ollama_client()
    connection = connect_to_postgres()

    try:
        results = retrieve_chunks(
            connection=connection,
            client=client,
            question=question,
            top_k=top_k,
        )

    finally:
        connection.close()

    print("\nRETRIEVAL RESULTS")
    print("=" * 75)

    print(f"Question: {question}")
    print(f"Top K: {top_k}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 75)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Page: {result['page_number']}"
        )

        print(
            f"Chunk index: "
            f"{result['chunk_index']}"
        )

        print("-" * 75)

        print(
            result["content"][:1_000]
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index a PDF or retrieve similar chunks."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help=(
            "Extract, chunk, embed and store the PDF."
        ),
    )

    index_parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete and recreate the Python vector "
            "table before indexing."
        ),
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Retrieve similar chunks.",
    )

    query_parser.add_argument(
        "question",
        nargs="+",
        help="The question to retrieve context for.",
    )

    query_parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    arguments = parse_arguments()

    try:
        if arguments.command == "index":
            index_document(
                reset=arguments.reset
            )

        elif arguments.command == "query":
            question = " ".join(
                arguments.question
            )

            run_query(
                question=question,
                top_k=arguments.top_k,
            )

    except Exception:
        logger.exception(
            "Vector-pipeline operation failed."
        )
        raise


if __name__ == "__main__":
    main()