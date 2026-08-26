from backend.vector_store import VectorStore
from backend.bm25 import BM25
from backend.reranker import Reranker

class HybridSearch:

    def __init__(self):

        print("Loading FAISS vector store...")

        self.vector_store = VectorStore()

        self.vector_store.load(
            index_path="data/faiss_index/index.faiss",
            metadata_path="data/faiss_index/chunks.json"
        )

        print("FAISS vector store loaded.")

        print("Loading BM25 index...")

        self.bm25 = BM25.load(
            "data/bm25_index.pkl"
        )

        print("BM25 index loaded.")

        print("Loading reranker...")

        self.reranker = Reranker()

        print("Hybrid search ready.")


    def search(
        self,
        query: str,
        retrieve_k: int = 20,
        top_k: int = 3
    ) -> list[dict]:

        # -------------------------
        # 1. FAISS semantic search
        # -------------------------

        vector_results = self.vector_store.search(
            query=query,
            top_k=retrieve_k
        )

        # -------------------------
        # 2. BM25 keyword search
        # -------------------------

        bm25_results = self.bm25.search(
            query=query,
            top_k=retrieve_k
        )

        # -------------------------
        # 3. Merge results
        # -------------------------

        candidates = {}

        for result in vector_results:

            chunk_id = result["chunk_id"]

            candidates[chunk_id] = result


        for chunk_index, score in bm25_results:

            chunk = self.vector_store.chunks[chunk_index]

            chunk_id = chunk["chunk_id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = chunk.copy()

        candidate_list = list(
            candidates.values()
        )

        # -------------------------
        # 4. Rerank candidates
        # -------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            chunks=candidate_list,
            top_k=top_k
        )

        return reranked_results


if __name__ == "__main__":

    hybrid_search = HybridSearch()

    queries = [

        "What is machine learning engineering?",

        "What is deep learning?",

        "How do neural networks learn?"
    ]

    for query in queries:

        print("\n" + "=" * 60)

        print("\nQuery:")
        print(query)

        results = hybrid_search.search(
            query=query,
            retrieve_k=20,
            top_k=3
        )

        print("\nReranked Results:")

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n--- Result {i} ---"
            )

            print(
                "Reranker Score:",
                round(
                    result["reranker_score"],
                    4
                )
            )

            print(
                f"Source: "
                f"{result['document_name']} "
                f"— Page "
                f"{result['page_number']}"
            )

            print()

            print(
                result["text"][:600]
            )