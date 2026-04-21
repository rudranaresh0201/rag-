# 🚀 RAGnosis — Ask With Evidence, Think With Structure

A full-stack Retrieval-Augmented Generation (RAG) system that lets users upload PDFs and ask questions with structured, context-grounded answers.

---

## ✨ Features

- 📄 Upload and query multiple PDFs  
- 🔍 Semantic search with vector embeddings  
- 🧠 LLM-powered structured responses (Summary, Key Points, Explanation)  
- 📊 Confidence scoring and evidence-backed answers  
- ⚡ FastAPI backend + React frontend  

---

## 📸 Screenshots

### 🧭 Landing Interface
![Landing](pics/landing.png)

---

### 📂 Document Upload & Sidebar
![Sidebar](pics/sidebar.png)

---

### 💬 Query + Structured Answer
![Answer](pics/answer.png)

---

## 🎥 Demo Video

Download and view: [Demo](pics/demo.mp4)

---

## ⚙️ Tech Stack

- **Frontend:** React (Vite)  
- **Backend:** FastAPI  
- **Vector DB:** ChromaDB  
- **Embeddings:** Sentence Transformers  
- **LLM:** TinyLlama (local inference)  

---

## 🚀 How to Run

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8004

# Frontend
cd frontend
npm install
npm run dev