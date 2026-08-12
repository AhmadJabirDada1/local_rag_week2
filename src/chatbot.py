from __future__ import annotations

import logging

from ollama import Client, ResponseError

from .config import (
    CHAT_MODEL,
    MIN_SIMILARITY,
    OLLAMA_HOST,
    TOP_K,
    VECTOR_TABLE,
    setup_logging,
)

from .vector_pipeline import (
    connect_to_postgres,
    retrieve_chunks,
)


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a document-grounded question-answering assistant.

Answer the user's question only using the retrieved
document context supplied below.

Do not use outside or pretrained knowledge to fill
missing information.

If the supplied context does not contain enough
information to answer the question, respond exactly:

I could not find this information in the indexed document.

Do not invent facts, sources, page numbers or quotations.
Give a clear and concise answer.
""".strip()


def build_context(
    retrieved_chunks: list[dict],
) -> str:
    """
    Combine retrieved chunks into one context string.
    """

    context_sections: list[str] = []

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        section = (
            f"[Source {rank}]\n"
            f"Document: {chunk['source']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Chunk: {chunk['chunk_index']}\n"
            f"Similarity: "
            f"{chunk['similarity']:.4f}\n\n"
            f"{chunk['content']}"
        )

        context_sections.append(section)

    return "\n\n---\n\n".join(
        context_sections
    )


def generate_answer(
    client: Client,
    question: str,
    context: str,
) -> str:
    """
    Send retrieved context and the question to Llama 3.2.
    """

    user_message = f"""
Retrieved document context:

{context}

User question:

{question}

Answer only from the retrieved document context.
""".strip()

    try:
        response = client.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            options={
                "temperature": 0.1,
            },
        )

    except ResponseError as exc:
        raise RuntimeError(
            f"Ollama chat request failed: "
            f"{exc.error}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            "Unable to generate an answer. Confirm "
            "that Ollama is running and llama3.2:3b "
            "is installed."
        ) from exc

    if hasattr(response, "message"):
        return response.message.content.strip()

    return response["message"]["content"].strip()


def display_sources(
    chunks: list[dict],
) -> None:
    """
    Display the chunks supplied to the LLM.
    """

    print("\nSources used:")

    for chunk in chunks:
        print(
            f"- {chunk['source']}, "
            f"page {chunk['page_number']}, "
            f"chunk {chunk['chunk_index']}, "
            f"similarity "
            f"{chunk['similarity']:.4f}"
        )


def main() -> None:
    setup_logging()

    ollama_client = Client(
        host=OLLAMA_HOST
    )

    connection = connect_to_postgres()

    try:
        row = connection.execute(
            f"""
            SELECT COUNT(*) AS row_count
            FROM {VECTOR_TABLE}
            """
        ).fetchone()

        row_count = int(
            row["row_count"]
        )

        if row_count == 0:
            raise RuntimeError(
                "The vector table is empty. Run "
                "'python -m src.vector_pipeline "
                "index --reset' first."
            )

        logger.info(
            "Chatbot started with %d indexed chunks.",
            row_count,
        )

        print("\nLOCAL PYTHON RAG CHATBOT")
        print("=" * 75)

        print(
            "Type 'exit' or 'quit' to stop."
        )

        while True:
            try:
                question = input(
                    "\nAsk a question: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):
                print("\nGoodbye.")
                break

            if question.lower() in {
                "exit",
                "quit",
            }:
                print("Goodbye.")
                break

            if not question:
                continue

            try:
                retrieved = retrieve_chunks(
                    connection=connection,
                    client=ollama_client,
                    question=question,
                    top_k=TOP_K,
                )

                if not retrieved:
                    print(
                        "\nI could not find this information "
                        "in the indexed document."
                    )
                    continue

                relevant_chunks = [
                    chunk
                    for chunk in retrieved
                    if float(
                        chunk["similarity"]
                    ) >= MIN_SIMILARITY
                ]

                if not relevant_chunks:
                    print(
                        "\nI could not find this information "
                        "in the indexed document."
                    )
                    continue

                context = build_context(
                    relevant_chunks
                )

                answer = generate_answer(
                    client=ollama_client,
                    question=question,
                    context=context,
                )

                print("\nAnswer:")
                print(answer)

                display_sources(
                    relevant_chunks
                )

            except Exception as exc:
                logger.exception(
                    "Question processing failed."
                )

                print(
                    f"\nAn error occurred: {exc}"
                )

    finally:
        connection.close()


if __name__ == "__main__":
    main()