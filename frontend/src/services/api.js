export const API_BASE = "http://127.0.0.1:8004";

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
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      throw new Error("Upload failed");
    }
    const data = await res.json();
    if (typeof onUploadProgress === "function") {
      onUploadProgress({ loaded: 1, total: 1 });
    }
    return data;
  } catch (error) {
    throw buildApiError(error, "Upload failed");
  }
}

export async function queryRag(query) {
  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) {
      throw new Error("Model is loading, please wait...");
    }
    const data = await res.json();
    return data;
  } catch (error) {
    throw buildApiError(error, "Model is loading, please wait...");
  }
}

export const queryApi = async (query) => {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const data = await res.json();
  return data;
};

export async function queryRagByDocument(query, documentId) {
  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        document_id: documentId,
      }),
    });
    if (!res.ok) {
      throw new Error("Model is loading, please wait...");
    }
    const data = await res.json();
    return data;
  } catch (error) {
    throw buildApiError(error, "Model is loading, please wait...");
  }
}

export async function listDocuments() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error("Failed to fetch documents");
    const data = await res.json();
    console.log("DOCUMENTS RESPONSE:", data);
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
  try {
    const res = await fetch(`${API_BASE}/reset`, {
      method: "DELETE",
    });
    if (!res.ok) {
      throw new Error("Failed to clear documents.");
    }
    const data = await res.json();
    return data;
  } catch (error) {
    throw buildApiError(error, "Failed to clear documents.");
  }
}
