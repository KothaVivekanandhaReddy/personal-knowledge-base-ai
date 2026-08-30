import time
import json
from pathlib import Path

from backend.vector_store import VectorStore
from backend.bm25 import BM25
from backend.reranker import Reranker

from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank
)


DATASET_PATH = (
    Path(__file__).parent
    / "evaluation_dataset.json"
)


def retrieve_candidates(
    vector_store,
    bm25,
    query,
    retrieve_k=20
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
    # 3. Merge candidates
    # -------------------------

    candidates = {}

    for result in vector_results:

        chunk_id = result["chunk_id"]

        candidates[chunk_id] = result.copy()

        candidates[chunk_id][
            "retrieved_by"
        ] = ["faiss"]

    for chunk_index, score in bm25_results:

        chunk = vector_store.chunks[
            chunk_index
        ]

        chunk_id = chunk["chunk_id"]

        if chunk_id not in candidates:

            candidates[chunk_id] = chunk.copy()

            candidates[chunk_id][
                "retrieved_by"
            ] = ["bm25"]

        else:

            candidates[chunk_id][
                "retrieved_by"
            ].append("bm25")

    return list(
        candidates.values()
    )


def main():

    print(
        "Loading evaluation dataset..."
    )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        evaluation_data = json.load(file)

    print(
        f"Loaded {len(evaluation_data)} "
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

    print(
        "Loading reranker..."
    )

    reranker = Reranker()

    recalls = []

    reciprocal_ranks = []

    latencies = []

    print(
        "\n" + "=" * 60
    )

    print(
        "FAISS + BM25 + RERANKER "
        "EVALUATION"
    )

    print(
        "=" * 60
    )

    for item in evaluation_data:

        query = item["query"]

        relevant_document = item[
            "relevant_documents"
        ]

        print(
            f"\nQuery: {query}"
        )

        start_time = time.perf_counter()

        candidates = retrieve_candidates(
            vector_store=vector_store,
            bm25=bm25,
            query=query,
            retrieve_k=20
        )

        results = reranker.rerank(
            query=query,
            chunks=candidates,
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

        recall = recall_at_k(results, relevant_document, k=5)
        rr = reciprocal_rank(results, relevant_document, k=5)


        recalls.append(recall)

        reciprocal_ranks.append(rr)

        latencies.append(latency)

        print(
            f"Expected: "
            f"{relevant_document}"
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
                f"{result.get('retrieved_by')}"
            )

            print(
                f"   Reranker Score: "
                f"{result['reranker_score']:.4f}"
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