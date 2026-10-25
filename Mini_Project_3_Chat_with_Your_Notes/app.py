from pathlib import Path
import io
import streamlit as st
from rag_core import extract_pdf_pages, chunk_pages, SemanticRetriever, extractive_answer, ollama_generate, NO_ANSWER

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "sample_study_notes.pdf"

st.set_page_config(page_title="Chat with Your Notes - Week 3", page_icon="📚", layout="wide")
st.title("Chat with Your Notes")
st.caption("Week 3 Mini Project 3 | PDF retrieval, page citations, and evidence-aware answers")
st.info("Default mode is a local extractive answer built only from retrieved PDF passages. Optional Ollama mode uses a local language model and does not require a paid API key.")

with st.sidebar:
    st.header("Document and retrieval settings")
    uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
    chunk_size = st.slider("Chunk size (words)", min_value=60, max_value=260, value=120, step=20)
    overlap = st.slider("Chunk overlap (words)", min_value=0, max_value=min(80, chunk_size-1), value=min(25, chunk_size//4), step=5)
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=8, value=4)
    backend_choice = st.selectbox("Embedding backend", ["auto (dense embeddings with TF-IDF fallback)", "TF-IDF only", "sentence-transformers only"])
    use_ollama = st.checkbox("Use Ollama local LLM for answer generation (optional)", value=False)
    ollama_model = st.text_input("Ollama model", value="llama3.2:3b", disabled=not use_ollama)
    threshold = st.slider("Minimum top-passage similarity", min_value=0.00, max_value=0.80, value=0.20, step=0.01)

source_label = uploaded.name if uploaded else DEFAULT_PDF.name
try:
    if uploaded:
        pages = extract_pdf_pages(io.BytesIO(uploaded.getvalue()))
    else:
        pages = extract_pdf_pages(DEFAULT_PDF)
    if not pages:
        st.error("No selectable text was extracted. Use a text-based PDF; scanned PDFs need OCR first.")
        st.stop()
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    backend = "tfidf" if backend_choice == "TF-IDF only" else ("sentence-transformer" if backend_choice == "sentence-transformers only" else "auto")
    # Cache the fitted index for the selected document/settings during this Streamlit process.
    cache_key = (source_label, len(chunks), chunk_size, overlap, backend, tuple((c.page, c.text[:80]) for c in chunks[:3]))
    if st.session_state.get("cache_key") != cache_key:
        index = SemanticRetriever(backend=backend).fit(chunks)
        st.session_state["cache_key"] = cache_key
        st.session_state["index"] = index
        st.session_state["chunks"] = chunks
    index = st.session_state["index"]
    if getattr(index, "warning", None):
        st.warning(index.warning)
    st.caption(f"Loaded **{source_label}** · {len(pages)} text pages · {len(chunks)} chunks · backend: **{index.backend}**")
except Exception as exc:
    st.error(f"Could not load or index the PDF: {exc}")
    st.stop()

question = st.text_input("Ask a question about your notes", placeholder="Example: What is precision?")
if st.button("Find answer", type="primary", disabled=not question.strip()):
    hits = index.search(question, top_k=top_k)
    if not hits or hits[0]["score"] < threshold:
        st.warning(NO_ANSWER)
        st.caption("No answer is shown because the highest similarity score did not pass your threshold. Similarity thresholds are heuristics; check the retrieved evidence before relying on an answer.")
    else:
        try:
            if use_ollama:
                answer = ollama_generate(question, hits, model_name=ollama_model)
                answer_pages = sorted({int(n) for n in __import__('re').findall(r'(?:PDF\s*)?p(?:age)?\.?\s*(\d+)', answer, flags=__import__('re').I)})
                if NO_ANSWER.lower() in answer.lower():
                    answer = NO_ANSWER
                    answer_pages = []
            else:
                answer, answer_pages = extractive_answer(question, hits)
            if answer.strip().lower().rstrip('.!') == NO_ANSWER.lower().rstrip('.!'):
                st.warning(NO_ANSWER)
            else:
                st.subheader("Answer")
                st.write(answer)
                visible_pages = answer_pages or sorted({r['chunk'].page for r in hits[:2]})
                st.markdown("**Source PDF pages:** " + ", ".join(str(p) for p in visible_pages))
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")
    with st.expander("Retrieved passages and similarity scores", expanded=True):
        for rank, hit in enumerate(hits, start=1):
            ch = hit["chunk"]
            st.markdown(f"**{rank}. PDF page {ch.page} · similarity {hit['score']:.3f} · chunk {ch.chunk_id}**")
            st.write(ch.text)
            st.divider()

with st.expander("How to test the mini project"):
    st.markdown("Use `test_questions.csv` in this folder. Ask answerable questions such as **What is precision?** and unsupported questions such as **What exact accuracy percentage did the example model achieve?** The unsupported question should return the fallback when evidence does not pass the selected threshold. Verify all answers against their source passages.")
