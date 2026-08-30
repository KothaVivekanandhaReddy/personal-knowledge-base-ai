import json
import time

from backend.vector_store import VectorStore

from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    mean
)


def load_evaluation_dataset():
    with open(
        "evaluation/evaluation_dataset.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():
    print("\nLoading evaluation dataset...")
    evaluation_data = load_evaluation_dataset()
    print(f"Loaded {len(evaluation_data)} evaluation queries.")

    print("\nLoading FAISS vector store...")
    vector_store = VectorStore()
    vector_store.load(
        index_path="data/faiss_index/index.faiss",
        metadata_path="data/faiss_index/chunks.json"
    )

    recall_scores = []
    mrr_scores = []
    latencies = []

    print("\n" + "=" * 60)
    print("FAISS ONLY EVALUATION")
    print("=" * 60)

    for item in evaluation_data:
        query = item["query"]
        relevant_document = item["relevant_documents"]

        print(f"\nQuery: {query}")

        start_time = time.perf_counter()
        results = vector_store.search(query=query, top_k=5)
        end_time = time.perf_counter()

        latency = end_time - start_time

        # FIX: Pass values as positional arguments to avoid keyword mismatches
        recall = recall_at_k(results, relevant_document, 5)
        mrr = reciprocal_rank(results, relevant_document, 5)

        recall_scores.append(recall)
        mrr_scores.append(mrr)
        latencies.append(latency)

        print(f"Expected: {relevant_document}")
        print(f"Recall@5: {recall}")
        print(f"MRR@5 contribution: {mrr:.4f}")
        print(f"Latency: {latency * 1000:.2f} ms")

        print("\nRetrieved documents:")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['document_name']} "
                f"(Page {result['page_number']})"
            )

    final_recall = mean(recall_scores)
    final_mrr = mean(mrr_scores)
    average_latency = mean(latencies)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nRecall@5: {final_recall:.4f}")
    print(f"MRR@5: {final_mrr:.4f}")
    print(f"Average Latency: {average_latency * 1000:.2f} ms")


if __name__ == "__main__":
    main()
