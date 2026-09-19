# 📄 PDF RAG Chat

A Retrieval-Augmented Generation (RAG) application that lets you upload PDFs through a web UI and ask questions about them.

Uploaded files are stored in SQLite, chunked and embedded, indexed in FAISS (dense) and BM25 (keyword) for hybrid retrieval, reranked with a cross-encoder, and answered by a local or hosted LLM with source citations.

This project was built as part of my AI Engineering learning journey to understand the fundamentals of Retrieval-Augmented Generation (RAG).

---

## ✨ Features

- 📤 Upload PDFs from the browser (Streamlit), stored in SQLite — no manual file placement needed
- ✂️ Automatic text chunking
- 🧠 Generate embeddings
- 🔀 Hybrid retrieval (FAISS dense + BM25 keyword search) with reciprocal rank fusion
- 🎯 Cross-encoder reranking
- 📚 Multiple PDFs, with per-document delete
- 📎 Source citations (file + page) with every answer
- 💬 Chat UI, plus a terminal CLI alternative

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain (text splitting)
- Hugging Face / Ollama
- FAISS + BM25 (rank_bm25)
- Sentence Transformers (embeddings + cross-encoder reranking)
- SQLite
- Python Dotenv

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/sharniks/pdf-rag-chat.git
cd pdf-rag-chat
```

### Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root.

```text
HF_API_TOKEN=your_huggingface_api_token
```

### Start the application

```bash
streamlit run app.py
```

Open the app in your browser, upload a PDF from the sidebar, and ask questions in the chat once it's ingested. Uploaded files are stored in `pdf_qna.db` (SQLite) along with their extracted chunks; the FAISS/BM25 indices live under `vectorstore/`.

### CLI alternative

You can also bulk-ingest PDFs placed in `data/` and chat from the terminal instead of the browser:

```bash
python ingest.py   # ingests every PDF in data/ into the same store the UI uses
python query.py     # terminal Q&A loop; type 'exit' to quit
```

```text
Ask a question (or type 'exit'):

> What is polymorphism?

Answer:
...
```

---

## 📂 Project Structure

```
pdf-rag-chat/
│
├── data/            # optional: PDFs for the CLI ingest path
├── vectorstore/      # generated FAISS + BM25 indices (gitignored)
├── pdf_qna.db        # SQLite: uploaded file bytes + chunk text (gitignored)
├── storage.py         # SQLite + FAISS/BM25 storage layer
├── ingest.py
├── query.py
├── app.py             # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 📚 Concepts Practiced

- Retrieval-Augmented Generation (RAG)
- Document Loading
- Text Chunking
- Embeddings
- Vector Databases (FAISS)
- Semantic Search
- Prompt Engineering

---

## 🚀 Future Improvements

- Multi-turn conversation memory (follow-up questions using prior chat context)
- Filter retrieval to a chosen subset of uploaded documents
- Retrieval/answer quality evaluation

---

## 👨‍💻 Author

**Nikhil**

Software Engineer transitioning into AI Engineering.
Building production-ready AI applications and sharing my learning journey through hands-on projects.
