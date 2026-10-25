from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from rag_core import extract_pdf_pages, chunk_pages, SemanticRetriever, extractive_answer, NO_ANSWER
pdf=ROOT/"Assignment_3_Document_Retrieval_Evaluation"/"sample_study_notes.pdf"
pages=extract_pdf_pages(pdf)
assert len(pages) >= 7, len(pages)
chunks=chunk_pages(pages, chunk_size=100, overlap=20)
retr=SemanticRetriever(backend="tfidf").fit(chunks)
hits=retr.search("How is precision calculated?", top_k=3)
assert hits and hits[0]["chunk"].page in range(1, len(pages)+1)
ans, pgs=extractive_answer("How is precision calculated?", hits)
assert ans and pgs
print(f"PASS: {len(pages)} text pages, {len(chunks)} chunks, backend={retr.backend}, top source page={hits[0]['chunk'].page}")
print("Answer sample:", ans)
