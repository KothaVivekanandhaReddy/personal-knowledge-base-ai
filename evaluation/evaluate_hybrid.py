import time
import json

from pathlib import Path

from backend.vector_store import VectorStore
from backend.bm25 import BM25

from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank
)


DATASET_PATH = (
    Path(__file__).parent
    / "evaluation_dataset.json"
)


with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as file:

    EVALUATION_DATA = json.load(file)


def hybrid_search(
    vector_store,
    bm25,
    query,
    retrieve_k=20,
    top_k=5,
    k=60
):

    # -------------------------
    # 1. FAISS retrieval
    # -------------------------

    vector_results = vector_store.search(
        query=query,
        top_k=retrieve_k
    )

    # -------------------------
    # 2. BM25 retrieval
    # -------------------------

    bm25_results = bm25.search(
        query=query,
        top_k=retrieve_k
    )

    # -------------------------
    # 3. Reciprocal Rank Fusion
    # -------------------------

    candidates = {}

    # FAISS ranking
    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        chunk_id = result["chunk_id"]

        if chunk_id not in candidates:

            candidates[chunk_id] = result.copy()

            candidates[chunk_id][
                "rrf_score"
            ] = 0

            candidates[chunk_id][
                "retrieved_by"
            ] = []

        candidates[chunk_id][
            "rrf_score"
        ] += 1 / (k + rank)

        candidates[chunk_id][
            "retrieved_by"
        ].append("faiss")

    # BM25 ranking
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

            candidates[chunk_id] = chunk.copy()

            candidates[chunk_id][
                "rrf_score"
            ] = 0

            candidates[chunk_id][
                "retrieved_by"
            ] = []

        candidates[chunk_id][
            "rrf_score"
        ] += 1 / (k + rank)

        candidates[chunk_id][
            "retrieved_by"
        ].append("bm25")

    # -------------------------
    # 4. Sort by fused score
    # -------------------------

    fused_results = sorted(
        candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return fused_results[:top_k]


def main():

    print(
        "Loading evaluation dataset..."
    )

    evaluation_data = EVALUATION_DATA

    print(
        f"Loaded "
        f"{len(evaluation_data)} "
        f"evaluation queries."
    )

    print(
        "\nLoading FAISS vector store..."
    )

    vector_store = VectorStore()

    vector_store.load(
        index_path=(
            "data/faiss_index/"
            "index.faiss"
        ),
        metadata_path=(
            "data/faiss_index/"
            "chunks.json"
        )
    )

    print(
        "Loading BM25 index..."
    )

    bm25 = BM25.load(
        "data/bm25_index.pkl"
    )

    recalls = []

    reciprocal_ranks = []

    latencies = []

    print(
        "\n" + "=" * 60
    )

    print(
        "FAISS + BM25 (RRF) "
        "EVALUATION"
    )

    print(
        "=" * 60
    )

    for item in evaluation_data:

        query = item["query"]

        expected_document = item[
            "relevant_documents"
        ]

        print(
            f"\nQuery: {query}"
        )

        start_time = time.perf_counter()

        results = hybrid_search(
            vector_store=vector_store,
            bm25=bm25,
            query=query,
            retrieve_k=20,
            top_k=5
        )

        latency = (
            time.perf_counter()
            - start_time
        ) * 1000

        retrieved_documents = [
            result["document_name"]
            for result in results
        ]

        recall = recall_at_k(
            results, #  Pass the raw results dictionary list
            expected_document,
            k=5
        )

        rr = reciprocal_rank(
            results, #  Pass the raw results dictionary list
            expected_document,
            k=5
        )

        recalls.append(recall)

        reciprocal_ranks.append(rr)

        latencies.append(latency)

        print(
            f"Expected: "
            f"{expected_document}"
        )

        print(
            f"Recall@5: "
            f"{recall}"
        )

        print(
            f"MRR@5 contribution: "
            f"{rr:.4f}"
        )

        print(
            f"Latency: "
            f"{latency:.2f} ms"
        )

        print(
            "\nRetrieved documents:"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['document_name']} "
                f"(Page "
                f"{result['page_number']})"
            )

            print(
                f"   Retrieved by: "
                f"{result['retrieved_by']}"
            )

            print(
                f"   RRF Score: "
                f"{result['rrf_score']:.6f}"
            )

    # -------------------------
    # Final metrics
    # -------------------------

    average_recall = (
        sum(recalls)
        / len(recalls)
    )

    average_mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nRecall@5: "
        f"{average_recall:.4f}"
    )

    print(
        f"MRR@5: "
        f"{average_mrr:.4f}"
    )

    print(
        f"Average Latency: "
        f"{average_latency:.2f} ms"
    )


if __name__ == "__main__":
    main()