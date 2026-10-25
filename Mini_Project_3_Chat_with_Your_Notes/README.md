# Mini Project 3 - Chat with Your Notes

## Features
- Load the included text-based PDF or upload another text-based PDF.
- Extract text while preserving 1-based PDF page numbers.
- Chunk each page with configurable chunk size and overlap.
- Embed chunks and questions using `sentence-transformers/all-MiniLM-L6-v2`; if the model cannot be loaded, `auto` mode falls back to TF-IDF vectors.
- Retrieve the top passages with similarity scores.
- Answer with sentences extracted from retrieved passages, with source page references.
- Optional Ollama local LLM generation; unsupported answers should return exactly `I could not find this in the document`.

## Run from the project root

```powershell
python -m streamlit run Mini_Project_3_Chat_with_Your_Notes/app.py
```

The terminal prints a local URL (usually `http://localhost:8501`). Open it in your browser. Select the PDF, ask a question, and inspect the retrieved evidence. The first dense-embedding use may download the model. For no model download, choose **TF-IDF only**.

## Optional local LLM (Ollama)
Install Ollama from its official website, make sure the local service is running, and pull a model, for example:

```powershell
ollama pull llama3.2:3b
```

Then check **Use Ollama local LLM for answer generation** in the app. This path uses a local model and no paid API. Ollama is optional; the default extractive mode works without it.

## Evaluation
- `test_questions.csv`: ten questions (eight answerable and two intentionally unsupported).
- `sample_answers_with_page_references.csv`: reference answers and expected source pages for verification.
- `sample_study_notes.pdf`: source document.

**Limitations:** Similarity scores are retrieval ranking signals, not proof of entailment. Adjust the threshold and verify the displayed passage and page citation before trusting a response. Scanned/image-only PDFs are not processed by OCR in this project.
