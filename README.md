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


---

### 📂 Document Upload & Sidebar
![Sidebar](pics/sidebar.png)

---

### 💬 Query + Structured Answer
![Answer](pics/answer.png)


### 🧭 Landing Interface
![Landing](pics/landing.png)

---

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