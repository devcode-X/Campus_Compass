# src/ingest.py
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import read_pdf, read_docx, read_text, clean_text, list_data_files



# === Project paths (always absolute) ===
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUT_DIR = ROOT_DIR / "data" / "processed"

# make sure output folder exists
OUT_DIR.mkdir(parents=True, exist_ok=True)


def file_to_text(path: str) -> str:
    """Read a single file and return its text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    elif path.suffix.lower() == ".docx":
        return read_docx(path)
    elif path.suffix.lower() == ".txt":
        return read_text(path)
    else:
        print(f"⚠️ Unsupported file type: {path.name}")
        return ""


def safe_filename(name: str) -> str:
    """Convert any filename to a Windows-safe version."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return safe[:150]  # keep below Windows 260-char path limit


def ingest_all():
    """Ingest every file from data/raw, chunk them, and save processed JSONs."""
    files = list_data_files(str(DATA_DIR))
    if not files:
        raise FileNotFoundError(f"❌ No documents found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )

    processed_docs = []

    for f in files:
        text = clean_text(file_to_text(f))
        if not text.strip():
            print(f"⚠️ Skipping empty file: {f}")
            continue

        chunks = splitter.split_text(text)
        docs = [
            Document(page_content=chunk, metadata={"source": Path(f).name, "chunk": i})
            for i, chunk in enumerate(chunks)
        ]

        # build safe path and ensure directory exists
        safe_name = safe_filename(Path(f).stem)
        out_path = OUT_DIR / f"{safe_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # write the preview JSON for inspection
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(
                "\n".join([
                    f"{d.metadata['source']}|{d.metadata['chunk']}|"
                    f"{d.page_content[:200].replace(chr(10), ' ')}"
                    for d in docs
                ])
            )

        processed_docs.extend(docs)
        print(f"✅ Processed {len(docs)} chunks from: {Path(f).name}")
        print(f"   → saved preview to {out_path}")

    print(f"\n📚 Total processed chunks: {len(processed_docs)}")
    return processed_docs


if __name__ == "__main__":
    docs = ingest_all()
    print(f"Ingested {len(docs)} chunks from {DATA_DIR}")
