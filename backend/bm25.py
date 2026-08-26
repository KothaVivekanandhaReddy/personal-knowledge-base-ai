import math
import pickle

from collections import Counter


class BM25:

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75
    ):

        self.documents = [
            self.tokenize(document)
            for document in documents
        ]

        self.k1 = k1
        self.b = b

        self.document_count = len(self.documents)

        self.document_lengths = [
            len(document)
            for document in self.documents
        ]

        self.average_document_length = (
            sum(self.document_lengths)
            / self.document_count
        )

        self.term_frequencies = [
            Counter(document)
            for document in self.documents
        ]

        self.document_frequencies = (
            self._calculate_document_frequencies()
        )

    def tokenize(self, text: str) -> list[str]:

        return text.lower().split()

    def _calculate_document_frequencies(self) -> dict:

        document_frequencies = {}

        for document in self.documents:

            unique_terms = set(document)

            for term in unique_terms:

                document_frequencies[term] = (
                    document_frequencies.get(term, 0) + 1
                )

        return document_frequencies

    def _idf(self, term: str) -> float:

        frequency = self.document_frequencies.get(term, 0)

        return math.log(
            (
                self.document_count
                - frequency
                + 0.5
            )
            /
            (
                frequency
                + 0.5
            )
            + 1
        )

    def score(
        self,
        query: str,
        document_index: int
    ) -> float:

        query_terms = self.tokenize(query)

        score = 0.0

        document_frequency = (
            self.term_frequencies[document_index]
        )

        document_length = (
            self.document_lengths[document_index]
        )

        for term in query_terms:

            if term not in document_frequency:
                continue

            term_frequency = (
                document_frequency[term]
            )

            idf = self._idf(term)

            numerator = (
                term_frequency
                * (self.k1 + 1)
            )

            denominator = (
                term_frequency
                +
                self.k1
                *
                (
                    1
                    - self.b
                    +
                    self.b
                    *
                    (
                        document_length
                        /
                        self.average_document_length
                    )
                )
            )

            score += idf * (
                numerator / denominator
            )

        return score

    def add_documents(
        self,
        documents: list[str]
    ):

        if not documents:

            print(
                "No new documents to add to BM25."
            )

            return

        print(
            f"Adding {len(documents)} "
            f"new documents to BM25..."
        )

        for document in documents:

            tokens = self.tokenize(
                document
            )

            self.documents.append(
                tokens
            )

            self.document_lengths.append(
                len(tokens)
            )

            self.term_frequencies.append(
                Counter(tokens)
            )

            unique_terms = set(
                tokens
            )

            for term in unique_terms:

                self.document_frequencies[
                    term
                ] = (

                    self.document_frequencies.get(
                        term,
                        0
                    )

                    + 1

                )

        self.document_count = len(
            self.documents
        )

        self.average_document_length = (

            sum(
                self.document_lengths
            )

            / self.document_count

        )

        print(
            f"BM25 now contains "
            f"{self.document_count} documents."
        )

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[tuple[int, float]]:

        scores = []

        for document_index in range(
            self.document_count
        ):

            score = self.score(
                query,
                document_index
            )

            scores.append(
                (
                    document_index,
                    score
                )
            )

        scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return scores[:top_k]

    def save(
        self,
        file_path: str
    ):

        with open(
            file_path,
            "wb"
        ) as file:

            pickle.dump(
                self,
                file
            )

        print(
            f"BM25 index saved to {file_path}"
        )

    @classmethod
    def load(
        cls,
        file_path: str
    ):

        with open(
            file_path,
            "rb"
        ) as file:

            bm25 = pickle.load(file)

        print(
            f"BM25 index loaded from {file_path}"
        )

        return bm25


if __name__ == "__main__":

    documents = [

        "Machine learning engineering combines machine learning and software engineering.",

        "Deep learning uses neural networks to learn complex patterns.",

        "Python functions are reusable blocks of code.",

        "Machine learning models learn patterns from data."

    ]

    bm25 = BM25(documents)

    bm25.save(
        "data/bm25_test.pkl"
    )

    loaded_bm25 = BM25.load(
        "data/bm25_test.pkl"
    )

    query = "machine learning engineering"

    results = loaded_bm25.search(
        query,
        top_k=3
    )

    print(f"\nQuery: {query}\n")

    for index, score in results:

        print(
            f"Score: {score:.4f}"
        )

        print(
            documents[index]
        )

        print()