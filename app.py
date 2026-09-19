import streamlit as st
import streamlit.components.v1 as components

import storage
import ingest
import query

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄")

storage.init_db()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📄 PDF RAG Chat")

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                doc_id = ingest.ingest_single_document(uploaded_file.name, file_bytes)
            if doc_id is None:
                st.info(f"'{uploaded_file.name}' already ingested, skipped.")
            else:
                st.success(f"Ingested '{uploaded_file.name}'.")

    st.divider()

    docs = storage.list_documents()
    if not docs:
        st.caption("No documents uploaded yet.")
    for doc in docs:
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{doc['filename']}**  \n{doc['chunk_count']} chunks")
        if col2.button("🗑️", key=f"delete_{doc['id']}"):
            storage.delete_document(doc["id"])
            st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    index = storage.load_or_init_faiss_index()
    bm25, chunk_ids = storage.load_bm25()

    dense_chunks = query.retrieve_context(index, question, k=15)
    bm25_chunks = query.bm25_search(bm25, chunk_ids, question, k=15)
    fused = query.reciprocal_rank_fusion(dense_chunks, bm25_chunks)

    if not fused:
        answer = "I don't know. No documents have been ingested yet."
        sources = []
    else:
        top_chunks = query.rerank(question, fused[:15], top_k=3)
        combined_context = "\n\n".join(c["text"] for c in top_chunks)

        if query.USE_LOCAL_LLM:
            answer = query.ask_ollama(question, combined_context)
        else:
            answer = query.ask_hf(question, combined_context)

        seen = set()
        sources = []
        for c in top_chunks:
            key = (c["source"], c["page"])
            if key not in seen:
                seen.add(key)
                sources.append(f"{c['source']}, page {c['page']}")

    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

# Streamlit doesn't auto-scroll the chat area as it grows, so nudge the
# browser to the bottom on every rerun. The message-count comment forces
# the injected script to actually re-run each time (Streamlit skips
# re-executing a components.html call whose content is unchanged).
components.html(
    f"""
    <script>
        const containers = [
            window.parent.document.querySelector('section.main'),
            window.parent.document.querySelector('section[data-testid="stMain"]'),
        ];
        containers.forEach((el) => {{ if (el) el.scrollTop = el.scrollHeight; }});
        window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
    </script>
    <!-- {len(st.session_state.history)} -->
    """,
    height=0,
)
