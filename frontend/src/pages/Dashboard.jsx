import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import AnimatedHeader from "../components/AnimatedHeader";
import ChatBox from "../components/ChatBox";
import Sidebar from "../components/Sidebar";
import {
  deleteDocument,
  listDocuments,
  queryApi,
  resetRag,
  uploadPdf,
} from "../services/api";

const MotionDiv = motion.div;
const MotionButton = motion.button;

function Dashboard() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeDocumentId, setActiveDocumentId] = useState(null);
  const [query, setQuery] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [querying, setQuerying] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState("");

  const inputDisabled = useMemo(
    () => uploading || querying || uploadedFiles.length === 0,
    [uploading, querying, uploadedFiles.length]
  );

  const pushMessage = (message) => {
    setMessages((prev) => [...prev, message]);
  };

  const refreshDocuments = async () => {
    const docs = await listDocuments();
    setUploadedFiles(docs);
  };

  useEffect(() => {
    let mounted = true;

    const bootstrap = async () => {
      try {
        const docs = await listDocuments();
        if (!mounted) {
          return;
        }
        setUploadedFiles(docs);
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

  const handleUpload = async (file) => {
    setError("");
    setUploading(true);
    setUploadProgress(0);

    try {
      const response = await uploadPdf(file, (event) => {
        if (!event.total) {
          setUploadProgress(50);
          return;
        }
        const value = Math.round((event.loaded * 100) / event.total);
        setUploadProgress(Math.min(100, value));
      });

      console.log("UPLOAD RESPONSE:", response);
      if (!response) {
        setError("Upload failed");
        return;
      }

      setError("");
      await refreshDocuments();
      setUploadProgress(100);
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
    console.log("SEND START");

    setQuerying(true);
    setError("");

    setMessages((prev) => [...prev, { role: "user", content: query }]);

    try {
      const response = await queryApi(query);

      console.log("RESPONSE RECEIVED:", response);

      const answer = response?.answer;
      const sources = Array.isArray(response?.sources) ? response.sources : [];

      console.log("ANSWER:", answer);

      if (answer && answer.length > 0) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: answer, sources },
        ]);
        console.log("MESSAGES STATE UPDATED");
      } else {
        setError("No response from model");
      }
    } catch (err) {
      console.error("QUERY ERROR:", err);
      setError("Query failed");
    } finally {
      console.log("FINALLY RUNNING");
      setQuerying(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-white text-slate-900">
      <MotionDiv
        animate={{ x: [0, 12, 0], y: [0, -8, 0] }}
        transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute -left-20 -top-20 h-72 w-72 rounded-full bg-blue-100/70 blur-[120px]"
      />
      <MotionDiv
        animate={{ x: [0, -12, 0], y: [0, 10, 0] }}
        transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute right-0 top-0 h-72 w-72 rounded-full bg-violet-100/45 blur-[120px]"
      />

      <div className="relative mx-auto grid min-h-screen max-w-[1500px] grid-cols-1 gap-5 p-4 md:grid-cols-[330px_1fr] md:p-6">
        <Sidebar
          uploadedFiles={uploadedFiles}
          activeDocumentId={activeDocumentId}
          uploading={uploading}
          uploadProgress={uploadProgress}
          onUpload={handleUpload}
          onSelectDocument={setActiveDocumentId}
          onDeleteFile={handleDeleteFile}
          onClearAll={handleClearAll}
          clearing={clearing}
        />

        <section className="flex min-h-[85vh] flex-col gap-4">
          <AnimatedHeader />

          {error && (
            <MotionDiv
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600"
            >
              {error}
            </MotionDiv>
          )}

          <div className="min-h-0 flex-1">
            <ChatBox key={messages.length} messages={messages} loading={querying} />
          </div>

          <div className="glass-card rounded-2xl p-3 md:p-4">
            <div className="flex items-end gap-3">
              <textarea
                rows={2}
                value={query}
                disabled={inputDisabled}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSend(query);
                  }
                }}
                placeholder="Ask your documents..."
                className="scrollbar-thin min-h-[52px] flex-1 resize-none rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
              />
              <MotionButton
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                type="button"
                onClick={() => handleSend(query)}
                disabled={inputDisabled || !query.trim()}
                className="h-[52px] rounded-xl bg-blue-500 px-5 text-sm font-semibold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-45"
              >
                Send
              </MotionButton>
            </div>
            {uploading && (
              <p className="mt-2 text-xs text-slate-500">Processing document...</p>
            )}
            {activeDocumentId && (
              <p className="mt-1 text-xs text-blue-600">Filtering answers to selected document.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

export default Dashboard;
