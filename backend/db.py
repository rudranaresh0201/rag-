from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_client = None
_embedder: SentenceTransformer | None = None


def get_client():
    global _client
    if _client is None:
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )
    return _client


def get_collection() -> Collection:
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedder()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()


def add_chunks(
    ids: List[str],
    chunks: List[str],
    metadatas: List[Dict[str, str | int]],
    embeddings: List[List[float]],
) -> None:
    collection = get_collection()
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)


def query_chunks(
    query_embedding: List[float],
    top_k: int,
    document_id: str | None = None,
) -> Dict:
    collection = get_collection()
    where = {"doc_id": document_id} if document_id else None
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def get_all_records() -> Dict:
    collection = get_collection()
    return collection.get(include=["metadatas", "documents"])


def delete_document(document_id: str) -> None:
    collection = get_collection()
    collection.delete(where={"doc_id": document_id})


def reset_database() -> None:
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(name=COLLECTION_NAME)
