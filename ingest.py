import fitz  # PyMuPDF library for reading PDF files
import os
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Configure logging so that INFO and ERROR messages are printed on console.
logging.basicConfig(level=logging.INFO)

# Path of the input PDF document
DATA_DIR = "./data"

# Location where FAISS vector index will be stored
INDEX_PATH = "./vectorstore/faiss_index"


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be read or extracted."""
    pass


def extract_pages_from_pdf(path):
    """
    Reads a PDF and returns a list of (page_num, page_text) tuples.
    Raises PDFExtractionError instead of silently returning None,
    so the caller can distinguish "missing file" from "corrupt PDF"
    - similar to throwing a checked FileNotFoundException vs an
    IOException in Java rather than returning null for both.
    """
    if not os.path.exists(path):
        raise PDFExtractionError(f"PDF file not found at path: {path}")

    try:
        pdf_document = fitz.open(path)
        pages = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pages.append((page_num + 1, page.get_text()))
        pdf_document.close()
        logging.info(f"Extracted {len(pages)} pages from {path}")
        return pages
    except Exception as e:
        raise PDFExtractionError(f"Error processing PDF {path}: {e}") from e

# -----------------------------------------------------------------------------
# Step 1 : Extract text from PDF
# -----------------------------------------------------------------------------


def extract_text_from_pdf(path):
    """
    Reads a PDF file and extracts text from every page.

    Parameters
    ----------
    path : str
        Path to the PDF document.

    Returns
    -------
    str
        Combined text from all pages.
        Returns None if the file does not exist or an error occurs.
    """

    # Verify that the PDF exists before attempting to open it.
    if os.path.exists(path):
        try:
            # Open PDF document
            pdf_document = fitz.open(path)

            # Stores text extracted from each page
            all_text = []

            # Iterate through every page in the PDF
            for page_num in range(len(pdf_document)):

                # Load one page at a time
                page = pdf_document.load_page(page_num)

                # Extract plain text from the page
                text = page.get_text()

                # Store page text
                all_text.append(text)

            # Always close the PDF after processing
            pdf_document.close()

            logging.info(
                f"PDF loaded successfully. Extracted {len(all_text)} pages."
            )

            # Merge all page texts into one large string
            return "\n".join(all_text)

        except Exception as e:
            logging.error(f"Error processing PDF: {e}")

    else:
        logging.error(f"PDF file not found at path: {path}")

    return None


# -----------------------------------------------------------------------------
# Step 2 : Split text into smaller chunks
# -----------------------------------------------------------------------------
def build_chunks_from_directory(data_dir=DATA_DIR, chunk_size=500, chunk_overlap=50):
    """
    Loops over every PDF in data_dir and returns a list of dicts:
    {"text": ..., "source": ..., "page": ...}
    This is your unit of retrieval from here on — not a bare string.
    Think of it like a small DTO/record instead of passing a raw String
    around and hoping everyone remembers what it represents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))

    if not pdf_paths:
        raise PDFExtractionError(f"No PDF files found in {data_dir}")

    for path in pdf_paths:
        pages = extract_pages_from_pdf(path)  # raises on failure, no silent None
        for page_num, page_text in pages:
            for chunk_text in splitter.split_text(page_text):
                all_chunks.append({
                    "text": chunk_text,
                    "source": os.path.basename(path),
                    "page": page_num,
                })

    logging.info(f"Total chunks created: {len(all_chunks)} from {len(pdf_paths)} PDF(s)")
    return all_chunks


# -----------------------------------------------------------------------------
# Step 3 : Generate embeddings
# -----------------------------------------------------------------------------
def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)


# -----------------------------------------------------------------------------
# Step 4 : Save embeddings into FAISS
# -----------------------------------------------------------------------------
def save_faiss_index(embeddings, chunks, index_path=INDEX_PATH):
    """
    Creates a FAISS index from embeddings and saves it to disk.

    Also stores a mapping between vector index and original text
    so retrieved vectors can be converted back into readable text.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Embedding vectors.

    chunks : list
        Original text chunks.

    index_path : str
        Destination path for FAISS index.
    """

    # Embedding dimension (e.g. 384 for MiniLM)
    dim = embeddings.shape[1]

    # Create a simple FAISS index using L2 (Euclidean) distance.
    # This index performs exact nearest-neighbor search.
    index = faiss.IndexFlatL2(dim)

    # Add all embedding vectors into the index
    index.add(embeddings)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Save FAISS index to disk
    faiss.write_index(index, index_path)

    # Save original text chunks separately.
    # FAISS stores vectors only, not the original text.
    with open(index_path + "_mapping.pkl", "wb") as f:
        pickle.dump(chunks, f)

    logging.info(f"FAISS index saved at {index_path}")


# Build once at ingest time, save alongside the FAISS index
def build_bm25_index(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    chunks = build_chunks_from_directory()

    embeddings = create_embeddings(chunks)
    logging.info(f"Embedding shape: {embeddings.shape}")

    save_faiss_index(np.array(embeddings), chunks)

    bm25 = build_bm25_index(chunks)
    with open(INDEX_PATH + "_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    logging.info("Vector database created successfully.")
