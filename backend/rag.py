from backend.vector_store import VectorStore
from backend.context_builder import ContextBuilder
from backend.prompt_builder import PromptBuilder
from backend.llm import LLM


class RAGPipeline:

    def __init__(self):

        print("Initializing FAISS vector store...")

        self.retriever = VectorStore()

        self.retriever.load(
            index_path="data/faiss_index/index.faiss",
            metadata_path="data/faiss_index/chunks.json"
        )

        print("Initializing context builder...")

        self.context_builder = ContextBuilder(
            max_chunks=3,
            max_context_characters=6000
        )

        print("Initializing prompt builder...")

        self.prompt_builder = PromptBuilder()

        print("Initializing LLM...")

        self.llm = LLM()

        print("RAG pipeline ready.")

    def query(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:

        results = self.retriever.search(
            query=question,
            top_k=top_k
        )
        similarity_threshold = 0.55

        if not results or results[0]["score"] < similarity_threshold:
            return {
                "answer": (
                    "I could not find the answer "
                    "in the provided documents."
                ),
                "sources": []
            }

        if not results:

            return {
                "answer": (
                    "I could not find the answer "
                    "in the provided documents."
                ),
                "sources": []
            }

        context = self.context_builder.build_context(
            results
        )

        prompt = self.prompt_builder.build(
            question=question,
            context=context
        )

        answer = self.llm.generate(
            prompt
        ).strip()


        sources = []
        seen_sources = set()

        context_results = results[:self.context_builder.max_chunks]
        for result in context_results:

            source_key = (
                result["document_name"],
                result["page_number"]
            )

            if source_key not in seen_sources:

                sources.append(
                    {
                        "document_name": (
                            result["document_name"]
                        ),
                        "page_number": (
                            result["page_number"]
                        )
                    }
                )

                seen_sources.add(
                    source_key
                )

        return {
            "answer": answer,
            "sources": sources
        }


if __name__ == "__main__":

    rag_pipeline = RAGPipeline()

    while True:

        question = input(
            "\nAsk a question "
            "(or type 'exit'): "
        )

        if question.lower() == "exit":
            break

        result = rag_pipeline.query(
            question
        )

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nSources:")

        if not result["sources"]:

            print("No sources found.")

        else:

            for source in result["sources"]:

                print(
                    f"- "
                    f"{source['document_name']} "
                    f"— Page "
                    f"{source['page_number']}"
                )