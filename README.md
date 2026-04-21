# DOCRAG

A Retrieval-Augmented Generation (RAG) system for querying technical PDF documents using local inference.

Driven engineering student focused on AI and automotive systems, building practical solutions using RAG, EV electronics, and scalable full-stack development.

---

## 🚀 Overview

DOCRAG is a full-stack RAG pipeline that:

- Converts PDF documents into vector embeddings  
- Retrieves semantically relevant context  
- Generates structured answers using a local LLM  

The system runs fully on-device — no external APIs, ensuring privacy and low-latency performance.

---

## 🏗️ Architecture

PDF → Chunking → Embeddings → ChromaDB → Retrieval → Context Filtering → LLM → Structured Answer

---

## 🧠 Features

- Upload PDF documents
- Automatic chunking and embedding
- Semantic search using vector database
- Context filtering and ranking
- LLM-based answer generation
- Structured output (Summary, Key Points, Explanation)
- Interactive React frontend

---

## ⚙️ Tech Stack

- Backend: FastAPI (Python)
- Embeddings: sentence-transformers (bge-base)
- Vector DB: ChromaDB
- LLM: TinyLlama (local inference)
- Frontend: React

---

## 📌 Example Queries

- Explain tractive system in electric vehicles  
- Explain AIR (Accumulator Isolation Relay) working  
- Explain EV system architecture  
- Summarize smart city infrastructure  
any query from any pdf user uploads

---

## 🧠 Key Contributions

- Built end-to-end RAG pipeline (PDF → retrieval → generation)
- Implemented custom retrieval filtering and ranking
- Reduced hallucinations using prompt engineering
- Designed context cleaning and sentence selection pipeline
- Integrated FastAPI backend with React frontend

---

## 🔮 Future Work

- Upgrade to larger models (Mistral, LLaMA)
- Add source citations
- Improve retrieval ranking
- Streaming responses

---

## 👨‍💻 Author

Rudra Naresh