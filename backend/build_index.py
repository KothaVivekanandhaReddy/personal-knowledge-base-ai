from pathlib import Path

from backend.ingestion import ingest_pdf
from backend.vector_store import VectorStore
from backend.bm25 import BM25


DOCUMENTS_DIRECTORY = "data/documents"

FAISS_INDEX_PATH = "data/faiss_index/index.faiss"
CHUNKS_PATH = "data/faiss_index/chunks.json"

BM25_INDEX_PATH = "data/bm25_index.pkl"          


def load_all_documents():

    documents_path = Path(DOCUMENTS_DIRECTORY)

    pdf_files = list(
        documents_path.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIRECTORY}"
        )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    all_chunks = []

    start_chunk_id = 0

    for pdf_file in pdf_files:

        print(
            f"Processing: {pdf_file.name}"
        )

        chunks = ingest_pdf(
            str(pdf_file),
            start_chunk_id=start_chunk_id
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        all_chunks.extend(chunks)

        start_chunk_id += len(chunks)

    return all_chunks


def main():

    print("Starting document ingestion...")

    all_chunks = load_all_documents()

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    # -------------------------
    # FAISS
    # -------------------------

    print("\nBuilding FAISS index...")

    vector_store = VectorStore()

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    vector_store.add_chunks(
        all_chunks
    )

    vector_store.save(
        index_path=FAISS_INDEX_PATH,
        metadata_path=CHUNKS_PATH
    )

    print(
        "FAISS index saved successfully."
    )

    # -------------------------
    # BM25
    # -------------------------

    print("\nBuilding BM25 index...")

    bm25 = BM25(
        texts
    )

    bm25.save(
        BM25_INDEX_PATH
    )

    print(
        "BM25 index saved successfully."
    )

    print(
        "\nIndex build complete."
    )


if __name__ == "__main__":
    main()