import json
from pathlib import Path

from backend.vector_store import VectorStore
from backend.bm25 import BM25


DATASET_PATH = (
    Path(__file__).parent
    / "evaluation_dataset.json"
)


def main():

    print("Loading evaluation dataset...")

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        evaluation_data = json.load(file)

    print(
        f"Loaded {len(evaluation_data)} evaluation queries."
    )

    print("\nLoading FAISS vector store...")

    vector_store = VectorStore()

    vector_store.load(
        index_path="data/faiss_index/index.faiss",
        metadata_path="data/faiss_index/chunks.json"
    )

    print("\nLoading BM25 index...")

    bm25 = BM25.load(
        "data/bm25_index.pkl"
    )

    output = []

    print(
        "\n" + "=" * 70
    )

    print(
        "CANDIDATE RELEVANCE GENERATION"
    )

    print(
        "=" * 70
    )

    for item in evaluation_data:

        query = item["query"]

        print(
            "\n" + "-" * 70
        )

        print(
            f"\nQUERY: {query}"
        )

        # -------------------------
        # FAISS candidates
        # -------------------------

        faiss_results = vector_store.search(
            query=query,
            top_k=10
        )

        # -------------------------
        # BM25 candidates
        # -------------------------

        bm25_results = bm25.search(
            query=query,
            top_k=10
        )

        candidates = {}

        # -------------------------
        # Add FAISS results
        # -------------------------

        for rank, result in enumerate(
            faiss_results,
            start=1
        ):

            chunk_id = result["chunk_id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_name": result[
                        "document_name"
                    ],
                    "page_number": result[
                        "page_number"
                    ],
                    "text": result.get(
                        "text",
                        result.get("chunk_text", "")
                    ),
                    "faiss_rank": None,
                    "bm25_rank": None
                }

            candidates[chunk_id][
                "faiss_rank"
            ] = rank

        # -------------------------
        # Add BM25 results
        # -------------------------

        for rank, (
            chunk_index,
            score
        ) in enumerate(
            bm25_results,
            start=1
        ):

            chunk = vector_store.chunks[
                chunk_index
            ]

            chunk_id = chunk["chunk_id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_name": chunk[
                        "document_name"
                    ],
                    "page_number": chunk[
                        "page_number"
                    ],
                    "text": chunk.get(
                        "text",
                        chunk.get("chunk_text", "")
                    ),
                    "faiss_rank": None,
                    "bm25_rank": None
                }

            candidates[chunk_id][
                "bm25_rank"
            ] = rank

        candidate_list = list(
            candidates.values()
        )

        # -------------------------
        # Print candidates
        # -------------------------

        print(
            "\nCANDIDATE CHUNKS:\n"
        )

        for index, candidate in enumerate(
            candidate_list,
            start=1
        ):

            print(
                f"{index}. "
                f"{candidate['document_name']}"
            )

            print(
                f"   Page: "
                f"{candidate['page_number']}"
            )

            print(
                f"   FAISS Rank: "
                f"{candidate['faiss_rank']}"
            )

            print(
                f"   BM25 Rank: "
                f"{candidate['bm25_rank']}"
            )

            print(
                "\n   TEXT:"
            )

            text = candidate["text"]

            print(
                text[:1000]
            )

            print(
                "\n" + "-" * 50
            )

        output.append(
            {
                "query": query,
                "candidates": candidate_list
            }
        )

    # -------------------------
    # Save candidate data
    # -------------------------

    with open(
        "evaluation/candidate_relevance.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nCandidate data saved to:"
    )

    print(
        "evaluation/candidate_relevance.json"
    )


if __name__ == "__main__":
    main()