from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from pypdf import PdfReader

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_PATH,
    setup_logging,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PageDocument:
    """
    Represents the extracted text from one PDF page.
    """

    text: str
    source: str
    page_number: int


@dataclass(frozen=True, slots=True)
class TextChunk:
    """
    Represents one searchable piece of document text.
    """

    text: str
    source: str
    page_number: int
    chunk_index: int


class PDFExtractionError(RuntimeError):
    """
    Raised when a PDF exists but usable text cannot be extracted.
    """


def clean_extracted_text(text: str) -> str:
    """
    Normalise whitespace produced during PDF extraction.

    This keeps paragraph breaks while removing repeated spaces
    and excessive blank lines.
    """

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    cleaned_lines: list[str] = []

    for line in text.splitlines():
        cleaned_line = " ".join(line.split())
        cleaned_lines.append(cleaned_line)

    cleaned_text = "\n".join(cleaned_lines)

    # Replace three or more newlines with two.
    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text,
    )

    return cleaned_text.strip()


def load_pdf(pdf_path: Path) -> list[PageDocument]:
    """
    Open a PDF and return one PageDocument for every page
    containing extractable text.
    """

    pdf_path = Path(pdf_path).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"The supplied PDF path is not a file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {pdf_path.name}"
        )

    logger.info("Opening PDF: %s", pdf_path)

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PDFExtractionError(
            f"Unable to open PDF: {exc}"
        ) from exc

    total_pages = len(reader.pages)

    logger.info(
        "Total PDF pages found: %d",
        total_pages,
    )

    page_documents: list[PageDocument] = []
    total_characters = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise PDFExtractionError(
                f"Text extraction failed on page "
                f"{page_number}: {exc}"
            ) from exc

        cleaned_text = clean_extracted_text(raw_text)

        if not cleaned_text:
            logger.warning(
                "Page %d contained no extractable text.",
                page_number,
            )
            continue

        page_document = PageDocument(
            text=cleaned_text,
            source=pdf_path.name,
            page_number=page_number,
        )

        page_documents.append(page_document)
        total_characters += len(cleaned_text)

        logger.debug(
            "Page %d extracted: %d characters",
            page_number,
            len(cleaned_text),
        )

    if not page_documents:
        raise PDFExtractionError(
            "No usable text was extracted from the PDF. "
            "The document may be scanned and require OCR."
        )

    logger.info(
        "Pages containing extractable text: %d",
        len(page_documents),
    )

    logger.info(
        "Total characters extracted: %d",
        total_characters,
    )

    return page_documents


def find_natural_boundary(
    text: str,
    start: int,
    maximum_end: int,
    minimum_end: int,
) -> int:
    """
    Try to finish a chunk near a natural paragraph,
    sentence or word boundary.
    """

    if maximum_end >= len(text):
        return len(text)

    separators = (
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
    )

    for separator in separators:
        position = text.rfind(
            separator,
            minimum_end,
            maximum_end,
        )

        if position != -1:
            return position + len(separator)

    return maximum_end


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split one string into overlapping character-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        maximum_end = min(
            start + chunk_size,
            text_length,
        )

        # Avoid cutting extremely early just to find a separator.
        minimum_end = min(
            start + int(chunk_size * 0.60),
            maximum_end,
        )

        end = find_natural_boundary(
            text=text,
            start=start,
            maximum_end=maximum_end,
            minimum_end=minimum_end,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - chunk_overlap

        # Safety against an infinite loop.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_chunks(
    pages: list[PageDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """
    Split every PDF page and preserve source/page information.
    """

    chunks: list[TextChunk] = []
    global_chunk_index = 0

    for page in pages:
        page_chunks = split_text(
            text=page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        logger.info(
            "Page %d produced %d chunks.",
            page.page_number,
            len(page_chunks),
        )

        for chunk_text in page_chunks:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=global_chunk_index,
                )
            )

            global_chunk_index += 1

    if not chunks:
        raise ValueError(
            "No chunks were created from the extracted pages."
        )

    chunk_lengths = [
        len(chunk.text)
        for chunk in chunks
    ]

    logger.info(
        "Total chunks created: %d",
        len(chunks),
    )

    logger.info(
        "Chunk lengths — minimum: %d | "
        "average: %.1f | maximum: %d",
        min(chunk_lengths),
        mean(chunk_lengths),
        max(chunk_lengths),
    )

    return chunks


def load_and_chunk_pdf(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[PageDocument], list[TextChunk]]:
    """
    Run the complete Wednesday document-processing pipeline.
    """

    pages = load_pdf(pdf_path)

    chunks = create_chunks(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return pages, chunks


def display_sample_chunks(
    chunks: list[TextChunk],
    sample_count: int,
) -> None:
    """
    Display a small number of chunks for inspection.
    """

    sample_count = max(0, sample_count)

    for sample_number, chunk in enumerate(
        chunks[:sample_count],
        start=1,
    ):
        print("\n" + "=" * 75)
        print(f"SAMPLE CHUNK {sample_number}")
        print("=" * 75)

        print(f"Source: {chunk.source}")
        print(f"Page number: {chunk.page_number}")
        print(f"Chunk index: {chunk.chunk_index}")
        print(f"Chunk length: {len(chunk.text)} characters")

        print("-" * 75)
        print(chunk.text)
        print("-" * 75)


def parse_arguments() -> argparse.Namespace:
    """
    Allow chunk settings to be changed from Terminal.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Extract text from a PDF and split it "
            "into overlapping chunks."
        )
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        default=PDF_PATH,
        help=f"PDF path. Default: {PDF_PATH}",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    arguments = parse_arguments()

    try:
        pages, chunks = load_and_chunk_pdf(
            pdf_path=arguments.pdf,
            chunk_size=arguments.chunk_size,
            chunk_overlap=arguments.chunk_overlap,
        )

        print("\nDOCUMENT INGESTION SUMMARY")
        print("=" * 75)

        print(
            f"PDF: {Path(arguments.pdf).resolve()}"
        )

        print(
            f"Pages with extracted text: {len(pages)}"
        )

        print(
            f"Total chunks: {len(chunks)}"
        )

        print(
            f"Configured chunk size: "
            f"{arguments.chunk_size}"
        )

        print(
            f"Configured chunk overlap: "
            f"{arguments.chunk_overlap}"
        )

        display_sample_chunks(
            chunks=chunks,
            sample_count=arguments.samples,
        )

    except Exception:
        logger.exception(
            "Document ingestion failed."
        )
        raise


if __name__ == "__main__":
    main()