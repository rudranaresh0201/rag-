import axios from "axios";

const BASE_URL = "http://127.0.0.1:8003";

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
});

function getBackendErrorMessage(payload, fallbackMessage) {
  if (!payload) {
    return fallbackMessage;
  }

  const detail = payload.detail;
  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object") {
    if (typeof detail.message === "string") {
      return detail.message;
    }
    if (typeof detail.error === "string") {
      return detail.error;
    }
  }

  if (typeof payload.message === "string") {
    return payload.message;
  }

  return fallbackMessage;
}

function buildApiError(error, fallbackMessage) {
  if (error?.code === "ECONNABORTED") {
    return new Error("Model is loading, please wait...");
  }

  const code = error?.response?.data?.detail?.code;
  if (code === "file_too_large") {
    return new Error("File too large. Max size: 200MB");
  }

  const backendMessage = getBackendErrorMessage(error?.response?.data, fallbackMessage);
  return new Error(backendMessage || fallbackMessage);
}

export async function uploadPdf(file, onUploadProgress) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await apiClient.post("/upload", formData, {
      onUploadProgress,
    });
    if (!response || !response.data) {
      throw new Error("Upload failed");
    }
    return response.data;
  } catch (error) {
    throw buildApiError(error, "Upload failed");
  }
}

export async function queryRag(query) {
  let response;
  try {
    response = await apiClient.post("/query", { query });
  } catch (error) {
    throw buildApiError(error, "Model is loading, please wait...");
  }
  return response.data;
}

export const queryApi = async (query) => {
  const res = await apiClient.post("/query", { query });
  return res.data;
};

export async function queryRagByDocument(query, documentId) {
  let response;
  try {
    response = await apiClient.post("/query", {
      query,
      document_id: documentId,
    });
  } catch (error) {
    throw buildApiError(error, "Model is loading, please wait...");
  }
  return response.data;
}

export async function listDocuments() {
  try {
    const res = await fetch("http://127.0.0.1:8003/documents");
    const data = await res.json();
    console.log("DOCUMENTS RAW:", data);
    return data;
  } catch (error) {
    throw buildApiError(error, "Failed to load documents.");
  }
}

export async function deleteDocument(documentId) {
  if (!documentId) return null;
  return null;
}

export async function resetRag() {
  let response;
  try {
    response = await apiClient.delete("/reset");
  } catch (error) {
    throw buildApiError(error, "Failed to clear documents.");
  }
  return response.data;
}
