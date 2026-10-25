# SkillNexis LLM Development - Week 3
## Embeddings, Semantic Search & RAG

This package includes Assignment 3 (Document Retrieval Evaluation) and Mini Project 3 (Chat with Your Notes). It uses the supplied text-based machine-learning study notes PDF.

## Project layout
- `rag_core.py`: shared PDF extraction, page-aware chunking, embeddings, similarity retrieval, extractive grounded answers, optional Ollama generation.
- `requirements.txt`: Python dependencies.
- `Assignment_3_Document_Retrieval_Evaluation/`: assignment notebook, sample PDF, ten questions, retrieval evaluation reference results, and expected answers.
- `Mini_Project_3_Chat_with_Your_Notes/`: Streamlit app, notebook, sample PDF, test questions, and source-reference answers.

## Quick setup in VS Code (Windows PowerShell)
1. Extract the ZIP file.
2. Open VS Code and choose **File > Open Folder...**. Select the extracted `SkillNexis_LLM_Development_Week_3` root folder (the folder containing `rag_core.py` and `requirements.txt`).
3. Open **Terminal > New Terminal** and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can avoid activation and run the venv executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. In VS Code, install/enable the **Python** and **Jupyter** extensions if needed. Open `Assignment_3_Document_Retrieval_Evaluation/assignment_3_document_retrieval_evaluation.ipynb`, choose the `.venv` kernel, and click **Run All**.
5. For the interactive mini project, in the terminal from the root folder run:

```powershell
python -m streamlit run Mini_Project_3_Chat_with_Your_Notes/app.py
```

6. Open the local URL displayed by Streamlit (usually `http://localhost:8501`). Ask questions and review the retrieved passages and source pages.

## Embedding modes
- `sentence-transformers/all-MiniLM-L6-v2` is the preferred dense embedding model. Its model files may download the first time you use it, so first-use requires internet access.
- `auto` mode falls back to TF-IDF vectors when dense embeddings cannot load. TF-IDF is a lightweight sparse-vector baseline and is less semantically expressive than contextual dense embeddings.
- Select **TF-IDF only** in the Streamlit app to avoid model downloads.

## Optional local LLM generation
The default app creates an extractive answer only from retrieved passages. To enable natural-language generation with a local model, install Ollama separately, run `ollama pull llama3.2:3b`, then check the Ollama option in the app. This is optional and uses no paid API key.

## Expected outputs
The assignment notebook writes `retrieval_evaluation_results.csv` and `chunk_size_comparison.csv`. The mini-project notebook writes `generated_test_answers.csv`. Precomputed TF-IDF references and expected answers are supplied so the expected CSV structure is visible before execution.

## Important limitations
Similarity scores are not proof of factual support. Similarity thresholds are heuristics. Always inspect source passages and page references. This project supports text-based PDFs; it does not perform OCR for scanned/image-only documents.
