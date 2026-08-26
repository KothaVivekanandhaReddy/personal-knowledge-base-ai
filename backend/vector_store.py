import os
import json

import faiss
import numpy as np

from backend.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        self.index = None

        self.chunks = []

        self.embedding_model = EmbeddingModel()


    def add_chunks(
        self,
        chunks: list[dict]
    ):

        if not chunks:

            return

        self.chunks = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("Generating embeddings...")

        embeddings = self.embedding_model.encode(
            texts
        )

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        print(
            "Creating FAISS cosine similarity index..."
        )

        faiss.normalize_L2(
            embeddings
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        print("Adding vectors to index...")

        self.index.add(
            embeddings
        )

        print(
            f"Successfully indexed "
            f"{self.index.ntotal} chunks."
        )


    def add_new_chunks(
        self,
        chunks: list[dict]
    ):

        if not chunks:

            print(
                "No new chunks to add."
            )

            return

        if self.index is None:

            print(
                "FAISS index is empty. "
                "Creating a new index..."
            )

            self.add_chunks(
                chunks
            )

            return

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(
            f"Generating embeddings for "
            f"{len(chunks)} new chunks..."
        )

        embeddings = self.embedding_model.encode(
            texts
        )

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        faiss.normalize_L2(
            embeddings
        )

        print(
            "Adding new vectors to "
            "existing FAISS index..."
        )

        self.index.add(
            embeddings
        )

        self.chunks.extend(
            chunks
        )

        print(
            f"FAISS index now contains "
            f"{self.index.ntotal} chunks."
        )


    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        if self.index is None:

            raise ValueError(
                "FAISS index has not been loaded."
            )

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:

                continue

            chunk = self.chunks[index]

            results.append(

                {
                    "chunk_id": (
                        chunk["chunk_id"]
                    ),

                    "text": (
                        chunk["text"]
                    ),

                    "document_name": (
                        chunk["document_name"]
                    ),

                    "page_number": (
                        chunk["page_number"]
                    ),

                    "score": float(score)
                }

            )

        return results


    def save(
        self,
        index_path: str = (
            "data/faiss_index/index.faiss"
        ),
        metadata_path: str = (
            "data/faiss_index/chunks.json"
        )
    ):

        if self.index is None:

            raise ValueError(
                "Cannot save an empty index."
            )

        os.makedirs(
            os.path.dirname(index_path),
            exist_ok=True
        )

        print(
            "Saving FAISS index..."
        )

        faiss.write_index(
            self.index,
            index_path
        )

        print(
            "Saving chunk metadata..."
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            "Vector store saved successfully."
        )


    def load(
        self,
        index_path: str = (
            "data/faiss_index/index.faiss"
        ),
        metadata_path: str = (
            "data/faiss_index/chunks.json"
        )
    ):

        if not os.path.exists(
            index_path
        ):

            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{index_path}"
            )

        if not os.path.exists(
            metadata_path
        ):

            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{metadata_path}"
            )

        print(
            "Loading FAISS index..."
        )

        self.index = faiss.read_index(
            index_path
        )

        print(
            "Loading chunk metadata..."
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(
                file
            )

        print(
            f"Loaded "
            f"{self.index.ntotal} chunks."
        )