from __future__ import annotations

from pathlib import Path

import pytest

from src.document_pipeline import (
    PageDocument,
    create_chunks,
    load_pdf,
    split_text,
)


def test_split_text_creates_multiple_chunks() -> None:
    text = (
        "Retrieval-Augmented Generation uses external "
        "documents to improve language-model answers. "
    ) * 30

    chunks = split_text(
        text=text,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert len(chunks) > 1


def test_chunks_do_not_exceed_maximum_size() -> None:
    text = "A" * 1_000

    chunks = split_text(
        text=text,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert all(
        len(chunk) <= 200
        for chunk in chunks
    )


def test_overlap_is_present() -> None:
    # No spaces are used, so the splitter must use the
    # exact character boundary.
    text = "".join(
        str(number % 10)
        for number in range(100)
    )

    chunks = split_text(
        text=text,
        chunk_size=30,
        chunk_overlap=5,
    )

    assert chunks[0][-5:] == chunks[1][:5]


def test_page_metadata_is_preserved() -> None:
    pages = [
        PageDocument(
            text=("Example page content. " * 30),
            source="test-document.pdf",
            page_number=7,
        )
    ]

    chunks = create_chunks(
        pages=pages,
        chunk_size=120,
        chunk_overlap=20,
    )

    assert len(chunks) > 0

    assert all(
        chunk.source == "test-document.pdf"
        for chunk in chunks
    )

    assert all(
        chunk.page_number == 7
        for chunk in chunks
    )


def test_invalid_overlap_raises_error() -> None:
    with pytest.raises(ValueError):
        split_text(
            text="Example text",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_missing_pdf_raises_error(
    tmp_path: Path,
) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(missing_pdf)