def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":
    text = (
        "Python is a programming language. "
        "It is widely used in machine learning. "
        "Python has a simple syntax. "
        "It supports many libraries."
    )

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(chunk)