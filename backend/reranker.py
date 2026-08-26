from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        print("Loading reranker model...")

        self.model = CrossEncoder(model_name)

        print("Reranker ready.")


    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3
    ) -> list[dict]:

        if not chunks:
            return []

        pairs = []

        for chunk in chunks:

            pairs.append(
                (
                    query,
                    chunk["text"]
                )
            )

        scores = self.model.predict(pairs)

        reranked_results = []

        for chunk, score in zip(chunks, scores):

            result = chunk.copy()

            result["reranker_score"] = float(score)

            reranked_results.append(result)

        reranked_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True
        )

        return reranked_results[:top_k]


if __name__ == "__main__":

    chunks = [

        {
            "text": (
                "Deep Reinforcement Learning "
                "is used to train agents."
            )
        },

        {
            "text": (
                "Deep learning is a subset of machine "
                "learning that uses neural networks with "
                "multiple layers."
            )
        },

        {
            "text": (
                "Python is a popular programming language."
            )
        }

    ]

    query = "What is deep learning?"

    reranker = Reranker()

    results = reranker.rerank(
        query=query,
        chunks=chunks,
        top_k=3
    )

    print("\nQuery:")
    print(query)

    print("\nReranked Results:")

    for result in results:

        print("\nScore:")
        print(
            round(
                result["reranker_score"],
                4
            )
        )

        print(result["text"])