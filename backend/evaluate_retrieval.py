import json
import time

from backend.hybrid_search import HybridSearch


EVALUATION_PATH = "data/evaluation/questions.json"

TOP_K = 5


def load_questions() -> list[dict]:

    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def reciprocal_rank(
    results: list[dict],
    expected_document: str
) -> float:

    for rank, result in enumerate(results, start=1):

        if result["document_name"] == expected_document:

            return 1 / rank

    return 0.0


def evaluate():

    print("Loading hybrid retriever...")

    retriever = HybridSearch()

    questions = load_questions()

    total_questions = len(questions)

    recall_hits = 0

    total_reciprocal_rank = 0.0

    total_latency = 0.0

    print(
        f"\nEvaluating {total_questions} questions...\n"
    )

    for item in questions:

        question_id = item["id"]

        question = item["question"]

        expected_document = item[
            "expected_document"
        ]

        print(f"Question {question_id}")

        print(f"Q: {question}")

        start_time = time.perf_counter()

        results = retriever.search(
            query=question,
            top_k=TOP_K
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        total_latency += latency

        retrieved_documents = []

        for result in results:

            document_name = result[
                "document_name"
            ]

            if document_name not in retrieved_documents:

                retrieved_documents.append(
                    document_name
                )

        hit = (
            expected_document
            in retrieved_documents
        )

        if hit:

            recall_hits += 1

        rr = reciprocal_rank(
            results,
            expected_document
        )

        total_reciprocal_rank += rr

        print(
            f"Expected: {expected_document}"
        )

        print(
            f"Retrieved: {retrieved_documents}"
        )

        print(
            f"Hit@{TOP_K}: {hit}"
        )

        print(
            f"Reciprocal Rank: {rr:.3f}"
        )

        print(
            f"Latency: {latency:.3f}s"
        )

        print("-" * 60)

    recall_at_k = (
        recall_hits / total_questions
    )

    mrr_at_k = (
        total_reciprocal_rank
        / total_questions
    )

    average_latency = (
        total_latency
        / total_questions
    )

    print("\n" + "=" * 60)

    print("RETRIEVAL EVALUATION RESULTS")

    print("=" * 60)

    print(
        f"Questions: {total_questions}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{recall_at_k:.3f}"
    )

    print(
        f"MRR@{TOP_K}: "
        f"{mrr_at_k:.3f}"
    )

    print(
        f"Average Latency: "
        f"{average_latency:.3f}s"
    )


if __name__ == "__main__":

    evaluate()