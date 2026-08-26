from pathlib import Path

from backend.pdf_loader import load_pdf
from backend.chunker import chunk_text


def ingest_pdf(
    pdf_path: str,
    start_chunk_id: int = 0
) -> list[dict]:

    pages = load_pdf(pdf_path)

    document_name = Path(pdf_path).name

    all_chunks = []

    chunk_id = start_chunk_id

    for page in pages:

        page_number = page["page_number"]

        page_text = page["text"]

        page_chunks = chunk_text(page_text)

        for chunk in page_chunks:

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": document_name,
                    "page_number": page_number,
                    "text": chunk
                }
            )

            chunk_id += 1

    return all_chunks


def ingest_documents(
    documents_directory: str
) -> list[dict]:

    documents_path = Path(documents_directory)

    all_chunks = []

    chunk_id = 0

    pdf_files = list(
        documents_path.glob("*.pdf")
    )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    for pdf_path in pdf_files:

        print(
            f"Processing: {pdf_path.name}"
        )

        chunks = ingest_pdf(
            str(pdf_path),
            start_chunk_id=chunk_id
        )

        all_chunks.extend(chunks)

        chunk_id += len(chunks)

        print(
            f"Created {len(chunks)} chunks."
        )

    return all_chunks


if __name__ == "__main__":

    chunks = ingest_documents(
        "data/documents"
    )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )