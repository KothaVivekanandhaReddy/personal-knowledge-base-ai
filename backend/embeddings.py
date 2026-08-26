from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size: int = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )


if __name__ == "__main__":
    model = EmbeddingModel()

    text = "Transformers use self-attention mechanisms."

    embedding = model.encode([text])

    print("Embedding shape:", embedding.shape)
    print("First 10 values:", embedding[0][:10])