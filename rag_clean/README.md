# PDF RAG Platform

Production-ready full-stack PDF Retrieval-Augmented Generation system with:

- FastAPI backend
- ChromaDB vector store
- Sentence Transformers embeddings (all-MiniLM-L6-v2)
- PyMuPDF ingestion
- Local Phi-3 generation (lazy-loaded)
- React + Vite premium frontend

## Architecture

### Backend

Path: backend/

- app.py: FastAPI routes, validation, CORS
- ingestion.py: PDF parsing and chunk/embedding ingestion
- retrieval.py: semantic retrieval and context building
- llm.py: lazy-loaded Phi-3 generation
- db.py: ChromaDB and embedding model access
- utils.py: text cleanup and chunking utilities

### Frontend

Path: frontend/

- Sidebar for document lifecycle controls
- Chat panel with animated responses
- Source-aware AI answers
- API integration for upload/query/reset and document management

## API Endpoints

- POST /upload
   - Upload PDF (5MB max)
   - Returns: status, chunks, doc_id, metadata
- GET /documents
   - Lists uploaded documents
- DELETE /document/{document_id}
   - Deletes one document from vector DB
- POST /query
   - Body: query, optional document_id, optional top_k
   - Returns grounded answer + sources
- DELETE /reset
   - Clears full vector database
- GET /health
   - Health check

## Run Locally

### 1) Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on Vite default port and calls backend at http://127.0.0.1:8000 unless overridden via VITE_API_BASE_URL.

