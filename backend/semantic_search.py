from embeddings import EmbeddingModel
from similarity import cosine_similarity


documents = [
    "Python is a high-level programming language.",
    "Transformers use self-attention mechanisms.",
    "Employees receive thirty days of annual leave.",
    "Deep learning models use neural networks.",
    "The company provides health insurance to employees.",
]


def search(query, documents, model, top_k=3):
    query_embedding = model.encode([query])[0]

    document_embeddings = model.encode(documents)

    results = []

    for document, document_embedding in zip(
        documents,
        document_embeddings
    ):
        score = cosine_similarity(
            query_embedding,
            document_embedding
        )

        results.append((document, score))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":
    model = EmbeddingModel()

    query = "How many vacation days do employees get?"

    results = search(
        query,
        documents,
        model,
        top_k=3
    )

    print("\nQuery:")
    print(query)

    print("\nResults:")

    for document, score in results:
        print(f"{score:.4f}  |  {document}")