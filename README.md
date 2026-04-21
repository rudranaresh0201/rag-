# 🚀 RAGnosis — Ask With Evidence, Think With Structure

# RAGnosis — AI Research Workspace

Hybrid Retrieval-Augmented Generation (RAG) system with keyword-aware reranking, multi-document querying, and structured LLM responses.

Built with FastAPI, React, ChromaDB, and Gemini / local LLM fallback.

---

## ✨ Features

- Hybrid retrieval (Vector + BM25)
- Keyword-aware reranking for improved relevance
- Structured responses (Summary, Key Points, Explanation)
- Multi-document PDF querying
- Local + API-based LLM support (Gemini / TinyLlama fallback) 

---

## 📸 Demo

### 🧠 Landing Page
![Landing](pics/landing.png)

### 📂 Document Management Sidebar
![Sidebar](pics/sidebar.png)

### 🧠 Structured Answer Output
![RAG Answer](pics/answer.png)

---

## 🎥 Demo Video

Due to GitHub size limits, the demo video is not stored in the repository.

👉 It can be shared separately upon request.



## 🎥 Demo Video

https://github.com/user-attachments/assets/8da7fa2f-c267-4468-af9b-e761287af932

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