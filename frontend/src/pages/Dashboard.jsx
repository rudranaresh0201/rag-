import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import {
  HiArrowTrendingUp,
  HiCheckBadge,
  HiDocumentText,
} from "react-icons/hi2";
import AnswerPanel from "../components/AnswerPanel";
import ChatLayout from "../components/ChatLayout";
import EvidencePanel from "../components/EvidencePanel";
import Sidebar from "../components/Sidebar";
import {
  API_BASE,
  deleteDocument,
  listDocuments,
  queryRagByDocument,
  queryApi,
  resetRag,
} from "../services/api";

const TABS = ["Answer", "Evidence", "Insights"];

function normalizeDocuments(payload) {
  return (payload?.documents || []).map((doc) => ({
    id: doc.doc_id,
    name: doc.filename,
    chunks: doc.chunks,
    size: doc.size,
    uploaded_at: doc.uploaded_at,
  }));
}

function normalizeSources(sources) {
  if (!Array.isArray(sources)) {
    return [];
  }

  return sources
    .map((source, index) => ({
      id: Number(source?.id) || index + 1,
      text: String(source?.text || "").trim(),
      document: String(source?.document || source?.file || "Uploaded Doc"),
      page:
        typeof source?.page === "number"
          ? source.page
          : Number.isFinite(Number(source?.page))
          ? Number(source.page)
          : null,
    }))
    .filter((item) => item.text.length > 0);
}

function confidenceFromSources(answer, sources) {
  const quality = Math.min(100, Math.round((answer.length / 900) * 100));
  const sourceStrength = Math.min(100, sources.length * 28);
  return Math.round(0.6 * sourceStrength + 0.4 * quality);
}

function Dashboard() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeDocumentId, setActiveDocumentId] = useState(null);
  const [query, setQuery] = useState("");
  const [activeTab, setActiveTab] = useState("Answer");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [querying, setQuerying] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState("");

  const inputDisabled = useMemo(() => uploading || querying, [uploading, querying]);

  const started = messages.length > 0;

  const latestAssistantMessage = useMemo(
    () => [...messages].reverse().find((item) => item.role === "assistant") || null,
    [messages]
  );

  const refreshDocuments = async () => {
    const data = await listDocuments();
    const normalized = normalizeDocuments(data);
    setUploadedFiles(normalized);
  };

  useEffect(() => {
    let mounted = true;

    const bootstrap = async () => {
      try {
        const data = await listDocuments();
        const normalized = normalizeDocuments(data);
        if (!mounted) {
          return;
        }
        setUploadedFiles(normalized);
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err?.message || "Failed to load documents.");
      }
    };

    bootstrap();

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    console.log("DOCUMENT STATE:", uploadedFiles);
  }, [uploadedFiles]);

  const handleUpload = async (file) => {
    setError("");
    setUploading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      setUploadProgress(100);
      const data = await response.json();
      console.log("UPLOAD RESPONSE:", data);

      if (!data?.doc_id && typeof data?.chunks !== "number") {
        throw new Error("Invalid response");
      }

      if (!data?.doc_id) {
        throw new Error("Invalid response");
      }

      setError("");
      await refreshDocuments();
      setUploadProgress(100);
      window.alert(`Upload successful: ${data.filename || file.name} (${data.chunks ?? 0} chunks)`);
    } catch (err) {
      setError(err?.message || "Upload failed");
    } finally {
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
      }, 250);
    }
  };

  const handleDeleteFile = async (fileId) => {
    if (!fileId) return;
    setError("");
    try {
      await deleteDocument(fileId);
      await refreshDocuments();
      if (activeDocumentId === fileId) {
        setActiveDocumentId(null);
      }
    } catch (err) {
      setError(err?.message || "Failed to delete document.");
    }
  };

  const handleClearAll = async () => {
    setError("");
    setClearing(true);
    try {
      await resetRag();
      setUploadedFiles([]);
      setActiveDocumentId(null);
      setMessages([]);
      setQuery("");
    } catch (err) {
      setError(err?.message || "Failed to clear documents.");
    } finally {
      setClearing(false);
    }
  };

  const handleSend = async (query) => {
    if (!query.trim()) {
      return;
    }

    setQuerying(true);
    setError("");
    setActiveTab("Answer");
    const ask = query.trim();
    setQuery("");

    setMessages((prev) => [...prev, { role: "user", content: ask }]);

    try {
      const response = activeDocumentId
        ? await queryRagByDocument(ask, activeDocumentId)
        : await queryApi(ask);

      const answer = String(response?.answer || "").trim();
      const sources = normalizeSources(response?.sources);
      const confidence = confidenceFromSources(answer, sources);

      if (answer && answer.length > 0) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: answer,
            sources,
            confidence,
            query: ask,
          },
        ]);
      } else {
        setError("No response from model");
      }
    } catch (err) {
      setError(err?.message || "Query failed");
    } finally {
      setQuerying(false);
    }
  };

  const sortedFiles = useMemo(
    () => [...uploadedFiles].sort((a, b) => String(b.uploaded_at).localeCompare(String(a.uploaded_at))),
    [uploadedFiles]
  );

  const sidebar = (
    <Sidebar
      uploadedFiles={sortedFiles}
      activeDocumentId={activeDocumentId}
      uploading={uploading}
      uploadProgress={uploadProgress}
      onUpload={handleUpload}
      onSelectDocument={setActiveDocumentId}
      onDeleteFile={handleDeleteFile}
      onClearAll={handleClearAll}
      clearing={clearing}
      collapsed={sidebarCollapsed}
      onToggleCollapsed={() => setSidebarCollapsed((current) => !current)}
    />
  );

  const history = (
    <div className="space-y-3">
      <AnimatePresence>
        {messages.map((message, index) => {
          const isUser = message.role === "user";
          return (
            <motion.div
              key={`${message.role}-${index}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`rounded-3xl border p-4 ${
                isUser
                  ? "ml-auto max-w-[85%] border-indigo-300/40 bg-indigo-400/10"
                  : "mr-auto max-w-[95%] border-white/15 bg-slate-950/45"
              }`}
            >
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.15em] text-slate-300">
                {isUser ? "You" : "Assistant"}
              </p>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">
                {message.content}
              </p>
            </motion.div>
          );
        })}
      </AnimatePresence>

      {querying && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mr-auto max-w-[95%] rounded-3xl border border-white/15 bg-slate-950/45 p-4"
        >
          <p className="text-xs uppercase tracking-[0.15em] text-slate-400">Assistant</p>
          <p className="mt-2 text-sm text-slate-200">Retrieving evidence and generating answer...</p>
        </motion.div>
      )}
    </div>
  );

  const latestSources = latestAssistantMessage?.sources || [];
  const latestAnswer = latestAssistantMessage?.content || "";
  const latestQuery = latestAssistantMessage?.query || "";
  const latestConfidence = latestAssistantMessage?.confidence || 0;

  const rightPanel = (
    <div className="flex h-full min-h-[600px] flex-col">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="inline-flex rounded-2xl border border-white/15 bg-slate-950/45 p-1">
          {TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`rounded-xl px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] transition ${
                activeTab === tab
                  ? "bg-indigo-500 text-white"
                  : "text-slate-300 hover:bg-white/10"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        <span className="rounded-full border border-emerald-300/30 bg-emerald-400/10 px-2 py-1 text-xs font-semibold text-emerald-100">
          Confidence {latestConfidence}%
        </span>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto pr-1">
        {activeTab === "Answer" && (
          <AnswerPanel answer={latestAnswer} query={latestQuery} loading={querying} />
        )}

        {activeTab === "Evidence" && <EvidencePanel sources={latestSources} query={latestQuery} />}

        {activeTab === "Insights" && (
          <div className="space-y-3">
            <div className="rounded-3xl border border-white/15 bg-white/5 p-4 backdrop-blur-xl">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Session Overview</p>
              <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-3">
                  <p className="text-xs text-slate-400">Questions Asked</p>
                  <p className="mt-1 text-2xl font-semibold text-slate-100">
                    {messages.filter((item) => item.role === "user").length}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-3">
                  <p className="text-xs text-slate-400">Evidence Cards</p>
                  <p className="mt-1 text-2xl font-semibold text-slate-100">{latestSources.length}</p>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-white/15 bg-white/5 p-4 backdrop-blur-xl">
              <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Signals</p>
              <ul className="mt-3 space-y-2 text-sm text-slate-200">
                <li className="inline-flex w-full items-center gap-2 rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-2">
                  <HiDocumentText className="text-indigo-300" />
                  Active documents: {uploadedFiles.length}
                </li>
                <li className="inline-flex w-full items-center gap-2 rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-2">
                  <HiCheckBadge className="text-emerald-300" />
                  Retrieval confidence: {latestConfidence}%
                </li>
                <li className="inline-flex w-full items-center gap-2 rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-2">
                  <HiArrowTrendingUp className="text-fuchsia-300" />
                  Current scope: {activeDocumentId ? "Single document" : "Cross-document"}
                </li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <motion.div
        animate={{ x: [0, 12, 0], y: [0, -8, 0] }}
        transition={{ duration: 16, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute -left-28 -top-24 h-[26rem] w-[26rem] rounded-full bg-indigo-600/30 blur-[140px]"
      />
      <motion.div
        animate={{ x: [0, -12, 0], y: [0, 10, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute right-0 top-0 h-[24rem] w-[24rem] rounded-full bg-fuchsia-500/20 blur-[140px]"
      />

      <div className="relative mx-auto max-w-[1640px] p-4 md:p-6">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 rounded-2xl border border-rose-300/35 bg-rose-400/10 px-4 py-3 text-sm text-rose-100"
          >
            {error}
          </motion.div>
        )}

        <ChatLayout
          started={started}
          query={query}
          onChangeQuery={setQuery}
          onSubmit={() => handleSend(query)}
          disabled={inputDisabled}
          loading={querying}
          canSubmit={uploadedFiles.length > 0}
          sidebar={sidebar}
          sidebarCollapsed={sidebarCollapsed}
          mobileSidebarOpen={mobileSidebarOpen}
          onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
          onCloseMobileSidebar={() => setMobileSidebarOpen(false)}
          history={history}
          rightPanel={rightPanel}
        />
      </div>
    </div>
  );
}

export default Dashboard;
