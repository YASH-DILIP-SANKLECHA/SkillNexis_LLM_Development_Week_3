# Assignment 3 - Document Retrieval Evaluation

## Assignment coverage
1. Extract text from `sample_study_notes.pdf`.
2. Split each page into chunks while preserving 1-based PDF page numbers.
3. Generate chunk/question vector embeddings and rank passages with cosine similarity.
4. Retrieve passages for ten test questions (eight answerable, two intentionally unanswerable).
5. Compare 100-word chunks with 180-word chunks (20-word overlap).
6. Record whether expected answer phrases occur in retrieved top-three passages.

## Files
- `assignment_3_document_retrieval_evaluation.ipynb`: complete runnable notebook.
- `sample_study_notes.pdf`: text-based seven-content-page PDF plus cover.
- `test_questions.csv`: test questions and expected evidence metadata.
- `sample_retrieval_results_tfidf.csv`: precomputed reference output for TF-IDF, so you can inspect the expected output structure before running.
- `sample_chunk_size_comparison_tfidf.csv`: reference chunk-size summary.
- `sample_answers_with_page_references.csv`: reference answers and expected pages.

## Run
Open the Week 3 root folder in VS Code and select the `.venv` Python interpreter. Install `requirements.txt` from the root folder. Open the notebook, then run all cells. The notebook writes `retrieval_evaluation_results.csv` and `chunk_size_comparison.csv` beside itself. The first sentence-transformer run may download model files. For a lightweight no-download baseline, change `backend="auto"` to `backend="tfidf"` in the notebook.

## Evaluation note
Expected evidence is checked using pre-written answer phrases. This makes the exercise reproducible, but it is not a full semantic entailment evaluation. Review the retrieved text, source pages, similarity scores, and the two unsupported queries manually.
