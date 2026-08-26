class PromptBuilder:

    def build(
        self,
        question: str,
        context: str
    ) -> str:

        prompt = f"""
You are a helpful AI assistant answering questions about
the user's documents.

Answer the question using ONLY the provided context.

Rules:
1. Do not use information outside the context.
2. If the answer cannot be found in the context, say:
   "I could not find the answer in the provided documents."
3. Do not invent information.
4. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

        return prompt.strip()


if __name__ == "__main__":

    context = """
[Source: ml_book.pdf — Page 10]
Machine learning engineering combines machine learning
and software engineering.
"""

    question = "What is machine learning engineering?"

    builder = PromptBuilder()

    prompt = builder.build(
        question=question,
        context=context
    )

    print(prompt)