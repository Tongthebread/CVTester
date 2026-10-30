from pathlib import Path

from langchain_core.documents import Document

MAX_BYTES = 10 * 1024 * 1024
MAX_CHARS = 100_000


def read_document(path: Path) -> list[Document]:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path.name}: maximum file size is 10 MB")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Use an unencrypted PDF")
        pages = [page.extract_text() or "" for page in reader.pages]
        if any(not page.strip() for page in pages):
            raise ValueError("PDF contains pages without extractable text; OCR or export to text first")
    elif suffix in {".txt", ".md"}:
        pages = [path.read_text(encoding="utf-8")]
    else:
        raise ValueError("Supported formats: text-based PDF, UTF-8 TXT, Markdown")
    if not any(page.strip() for page in pages):
        raise ValueError(f"{path.name}: document is empty")
    if sum(map(len, pages)) > MAX_CHARS:
        raise ValueError(f"{path.name}: maximum extracted length is {MAX_CHARS:,} characters")
    return [Document(page_content=text, metadata={"source": path.name, "page": i})
            for i, text in enumerate(pages, 1)]


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = []
    for document in documents:
        # shortcut: overlapping character windows; use layout-aware parsing for complex CVs.
        for start in range(0, len(document.page_content), 800):
            text = document.page_content[start:start + 1000]
            if text.strip():
                chunks.append(Document(page_content=text, metadata={
                    **document.metadata, "chunk_id": f"cv-{len(chunks) + 1}", "offset": start,
                }))
    if not chunks:
        raise ValueError("No CV text to index")
    return chunks
