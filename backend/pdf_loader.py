import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    document = pymupdf.open(file_path)
    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = "data/sample.pdf"

    pages = load_pdf(pdf_path)

    print(f"Pages extracted: {len(pages)}")

    for page in pages[:2]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:1000])