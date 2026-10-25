"""Core document retrieval functions for SkillNexis Week 3.

Default backend: sentence-transformers/all-MiniLM-L6-v2 (downloaded on first use).
Fallback backend: TF-IDF vectors, useful where model downloads are unavailable.
"""
from __future__ import annotations
import io, re, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NO_ANSWER = "I could not find this in the document"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class PageText:
    page: int
    text: str
    source: str = "sample_study_notes.pdf"

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    source: str
    start_word: int
    end_word: int


def extract_pdf_pages(source) -> list[PageText]:
    """Extract text from a path, bytes object, or file-like object. Page numbers are 1-based PDF pages."""
    if isinstance(source, (str, Path)):
        reader = PdfReader(str(source))
        source_name = Path(source).name
    elif isinstance(source, bytes):
        reader = PdfReader(io.BytesIO(source))
        source_name = "uploaded_document.pdf"
    else:
        reader = PdfReader(source)
        source_name = getattr(source, "name", "uploaded_document.pdf")
    result = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            result.append(PageText(page=index, text=text, source=source_name))
    return result


def chunk_pages(pages: Sequence[PageText], chunk_size: int = 120, overlap: int = 25) -> list[Chunk]:
    """Split each PDF page separately to preserve exact source page metadata."""
    if chunk_size < 20:
        raise ValueError("chunk_size should be at least 20 words for this exercise")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    chunks=[]
    for page_obj in pages:
        words = page_obj.text.split()
        if not words:
            continue
        start=0; part=0
        while start < len(words):
            end=min(start+chunk_size, len(words))
            text=" ".join(words[start:end]).strip()
            if text:
                chunks.append(Chunk(f"p{page_obj.page}-c{part}", text, page_obj.page, page_obj.source, start, end))
            if end >= len(words):
                break
            start=end-overlap
            part += 1
    return chunks

class SemanticRetriever:
    """Similarity retriever with contextual sentence embeddings and a TF-IDF fallback."""
    def __init__(self, backend: str = "auto"):
        if backend not in {"auto", "sentence-transformer", "tfidf"}:
            raise ValueError("backend must be 'auto', 'sentence-transformer', or 'tfidf'")
        self.requested_backend=backend
        self.backend=None
        self.model=None
        self.vectorizer=None
        self.matrix=None
        self.chunks=[]
        self.warning=None

    def fit(self, chunks: Sequence[Chunk]):
        if not chunks:
            raise ValueError("No chunks were created. Check that the PDF contains extractable text.")
        self.chunks=list(chunks)
        if self.requested_backend != "tfidf":
            try:
                from sentence_transformers import SentenceTransformer
                self.model=SentenceTransformer(MODEL_NAME)
                self.matrix=self.model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
                self.backend="sentence-transformer"
                return self
            except Exception as exc:
                if self.requested_backend == "sentence-transformer":
                    raise RuntimeError(f"Could not load {MODEL_NAME}. Check internet access and install sentence-transformers. Details: {exc}") from exc
                self.warning=f"Dense embedding model unavailable; using TF-IDF fallback. Details: {type(exc).__name__}: {exc}"
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2), lowercase=True, strip_accents="unicode", sublinear_tf=True)
        self.matrix=self.vectorizer.fit_transform([c.text for c in chunks])
        self.backend="tfidf"
        return self

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self.matrix is None:
            raise RuntimeError("Call fit(chunks) before search(query).")
        if not query.strip():
            return []
        if self.backend == "sentence-transformer":
            q=self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            scores=(self.matrix @ q[0]).astype(float)
        else:
            q=self.vectorizer.transform([query])
            scores=cosine_similarity(q, self.matrix).ravel()
        order=np.argsort(-scores)[:max(1, min(int(top_k), len(self.chunks)))]
        return [{"chunk": self.chunks[int(i)], "score": float(scores[int(i)])} for i in order]


def extractive_answer(query: str, retrieved: Sequence[dict], max_sentences: int = 3) -> tuple[str, list[int]]:
    """Return an extractive, passage-grounded answer without an external LLM."""
    candidates=[]
    seen=set()
    for item in retrieved:
        ch=item["chunk"]
        # Keep a sentence boundary near punctuation; PDF line breaks may already be flattened.
        for sentence in re.split(r'(?<=[.!?])\s+', ch.text):
            sentence=re.sub(r'\s+', ' ', sentence).strip()
            if len(sentence) < 25 or sentence.lower() in seen:
                continue
            seen.add(sentence.lower())
            candidates.append((sentence, ch.page))
    if not candidates:
        return NO_ANSWER, []
    # Score query against candidate sentences with a small lexical model. Candidate sentences
    # are selected only from retrieved PDF chunks, which keeps this mode grounded.
    corpus=[query] + [s for s,_ in candidates]
    vec=TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    try:
        mat=vec.fit_transform(corpus)
        sims=cosine_similarity(mat[0:1], mat[1:]).ravel()
    except ValueError:
        sims=np.zeros(len(candidates))
    ranked=sorted(zip(sims, candidates), key=lambda item: item[0], reverse=True)
    selected=[]
    pages=[]
    for score, (sentence, page) in ranked:
        if sentence in [x[0] for x in selected]:
            continue
        selected.append((sentence, float(score)))
        pages.append(page)
        if len(selected) >= max_sentences:
            break
    answer=" ".join(sentence for sentence,_ in selected)
    return answer or NO_ANSWER, sorted(set(pages))


def ollama_generate(query: str, retrieved: Sequence[dict], model_name: str = "llama3.2:3b", timeout: int = 90) -> str:
    """Optional generation through a locally running Ollama server; no paid API is used."""
    import requests
    context="\n\n".join(f"[PDF page {r['chunk'].page}] {r['chunk'].text}" for r in retrieved)
    prompt=("Answer the question using ONLY the source passages below. Do not use outside knowledge. "
            f"Cite supporting pages as [PDF p. N]. If the passages do not contain the answer, reply exactly: '{NO_ANSWER}'.\n\n"
            f"SOURCE PASSAGES:\n{context}\n\nQUESTION: {query}\nANSWER:")
    response=requests.post("http://localhost:11434/api/generate", json={"model":model_name,"prompt":prompt,"stream":False}, timeout=timeout)
    response.raise_for_status()
    answer=str(response.json().get("response", "")).strip()
    return answer or NO_ANSWER
