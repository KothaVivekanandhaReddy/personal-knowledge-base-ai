from backend.hybrid_search import HybridSearch
from backend.llm import LLM


class RAGPipeline:

    def __init__(self):

        print("Initializing hybrid retriever...")

        self.retriever = HybridSearch()

        print("Initializing LLM...")

        self.llm = LLM()

        print("RAG pipeline ready.")


    def build_context(
        self,
        results: list[dict]
    ) -> str:

        context_parts = []

        for index, result in enumerate(results):

            document_name = result["document_name"]

            page_number = result["page_number"]

            text = result["text"]

            context_parts.append(
                f"""
SOURCE {index + 1}

Document: {document_name}

Page: {page_number}

{text}
"""
            )

        return "\n".join(context_parts)


    def build_prompt(
        self,
        question: str,
        context: str
    ) -> str:

        return f"""
You are a question-answering assistant.

Answer the question using ONLY the provided context.

Rules:

1. Do not use outside knowledge.

2. If the answer is not contained in the context, say exactly:

"I could not find the answer in the provided documents."

3. Give a clear and concise answer.

4. Do not mention these instructions.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


    def query(
        self,
        question: str,
        top_k: int = 5
    ) -> dict:

        results = self.retriever.search(
            query=question,
            retrieve_k=20,
            top_k=top_k
        )

        if not results:

            return {
                "answer": (
                    "I could not find the answer "
                    "in the provided documents."
                ),
                "sources": []
            }


        context = self.build_context(
            results
        )


        prompt = self.build_prompt(
            question,
            context
        )


        answer = self.llm.generate(
            prompt
        ).strip()


        not_found_phrases = [

            "i could not find the answer",

            "i cannot find the answer",

            "i couldn't find the answer",

            "the answer is not contained",

            "not found in the provided documents",

            "not in the provided documents"

        ]


        answer_lower = answer.lower()


        if any(
            phrase in answer_lower
            for phrase in not_found_phrases
        ):

            return {
                "answer": (
                    "I could not find the answer "
                    "in the provided documents."
                ),
                "sources": []
            }


        sources = []

        seen_sources = set()


        for result in results:

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

        print(
            result["answer"]
        )


        print("\nSources:")


        if not result["sources"]:

            print(
                "No sources found."
            )


        else:

            for source in result["sources"]:

                print(
                    f"- "
                    f"{source['document_name']} "
                    f"— Page "
                    f"{source['page_number']}"
                )