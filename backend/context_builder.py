class ContextBuilder:

    def __init__(
        self,
        max_chunks: int = 3, #Prevents sending too many documents
        max_context_characters: int = 6000 #Prevents the prompt from becoming too large.
    ):
        self.max_chunks = max_chunks
        self.max_context_characters = (
            max_context_characters
        )

    def build_context(
        self,
        results: list[dict]
    ) -> str:

        context_parts = []

        current_length = 0

        for result in results[:self.max_chunks]:

            text = result["text"]

            source = (
                f'{result["document_name"]} '
                f'— Page {result["page_number"]}'
            )

            chunk = (
                f"[Source: {source}]\n"
                f"{text}"
            )

            if (
                current_length + len(chunk)
                > self.max_context_characters
            ):
                break

            context_parts.append(chunk)

            current_length += len(chunk)

        return "\n\n".join(context_parts)
    
if __name__ == "__main__":

    results = [
        {
            "document_name": "ml_book.pdf",
            "page_number": 10,
            "text": (
                "Machine learning engineering combines "
                "machine learning and software engineering."
            )
        },
        {
            "document_name": "dl_book.pdf",
            "page_number": 50,
            "text": (
                "Deep learning is based on neural networks."
            )
        }
    ]

    builder = ContextBuilder()

    context = builder.build_context(results)

    print("\nContext:\n")

    print(context)