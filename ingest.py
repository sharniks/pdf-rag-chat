import fitz  # PyMuPDF library for reading PDF files
import os
import glob
import logging
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

import storage

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Configure logging so that INFO and ERROR messages are printed on console.
logging.basicConfig(level=logging.INFO)

# Directory scanned by the CLI entry point below for bulk (re-)ingestion.
DATA_DIR = "./data"


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be read or extracted."""
    pass


def _extract_pages(pdf_document):
    pages = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pages.append((page_num + 1, page.get_text()))
    return pages


def extract_pages_from_pdf(path):
    """
    Reads a PDF from disk and returns a list of (page_num, page_text) tuples.
    Raises PDFExtractionError instead of silently returning None, so the
    caller can distinguish "missing file" from "corrupt PDF".
    """
    if not os.path.exists(path):
        raise PDFExtractionError(f"PDF file not found at path: {path}")

    try:
        pdf_document = fitz.open(path)
        pages = _extract_pages(pdf_document)
        pdf_document.close()
        logging.info(f"Extracted {len(pages)} pages from {path}")
        return pages
    except Exception as e:
        raise PDFExtractionError(f"Error processing PDF {path}: {e}") from e


def extract_pages_from_bytes(file_bytes, filename="<uploaded>"):
    """Same as extract_pages_from_pdf, but for in-memory PDF bytes (e.g. an upload)."""
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        pages = _extract_pages(pdf_document)
        pdf_document.close()
        logging.info(f"Extracted {len(pages)} pages from {filename}")
        return pages
    except Exception as e:
        raise PDFExtractionError(f"Error processing PDF {filename}: {e}") from e


# -----------------------------------------------------------------------------
# Step 2 : Split text into smaller chunks
# -----------------------------------------------------------------------------
def chunk_pages(pages, chunk_size=500, chunk_overlap=50):
    """
    Splits a list of (page_num, page_text) tuples into {"page", "text"} dicts.
    This is the unit of retrieval from here on — not a bare string.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for page_num, page_text in pages:
        for chunk_text in splitter.split_text(page_text):
            chunks.append({"page": page_num, "text": chunk_text})
    return chunks


# -----------------------------------------------------------------------------
# Step 3 : Generate embeddings
# -----------------------------------------------------------------------------
def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)


# -----------------------------------------------------------------------------
# Step 4 : Ingest one document end-to-end (used by both the CLI below and app.py)
# -----------------------------------------------------------------------------
def ingest_single_document(filename, file_bytes):
    """
    Stores the document's bytes, extracts/chunks/embeds its text, and adds
    the chunks + embeddings to the shared storage layer.

    Returns the new document id, or None if this exact file was already
    ingested (detected by content hash in storage.add_document).
    """
    doc_id = storage.add_document(filename, file_bytes)
    if doc_id is None:
        logging.info(f"Skipping duplicate file: {filename}")
        return None

    pages = extract_pages_from_bytes(file_bytes, filename)
    chunks = chunk_pages(pages)
    if chunks:
        embeddings = create_embeddings(chunks)
        storage.add_chunks_and_embed(doc_id, chunks, embeddings)

    logging.info(f"Ingested {filename}: {len(chunks)} chunks")
    return doc_id


# -----------------------------------------------------------------------------
# Main Execution — bulk-ingest every PDF in DATA_DIR
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    storage.init_db()

    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_paths:
        raise PDFExtractionError(f"No PDF files found in {DATA_DIR}")

    for path in pdf_paths:
        with open(path, "rb") as f:
            file_bytes = f.read()
        ingest_single_document(os.path.basename(path), file_bytes)

    logging.info("Ingestion complete.")
