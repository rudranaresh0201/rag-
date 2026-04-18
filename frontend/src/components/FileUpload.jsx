import { motion } from "framer-motion";
import { useRef, useState } from "react";
import { HiArrowUpTray, HiMiniArrowPath } from "react-icons/hi2";

function FileUpload({ onUpload, uploading, uploadProgress, collapsed = false }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = (files) => {
    const file = files?.[0];
    if (!file) {
      return;
    }
    onUpload(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    handleFiles(event.dataTransfer.files);
  };

  return (
    <motion.div
      whileHover={{ y: -1 }}
      onDragOver={(event) => {
        event.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
      className={`rounded-2xl border p-3 transition ${
        dragActive
          ? "border-indigo-300/60 bg-indigo-400/10"
          : "border-white/15 bg-slate-900/45"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(event) => handleFiles(event.target.files)}
        disabled={uploading}
      />

      <button
        type="button"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-indigo-300/35 bg-indigo-500/15 px-3 py-2 text-xs font-semibold uppercase tracking-[0.13em] text-indigo-100 transition hover:bg-indigo-500/25 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {uploading ? <HiMiniArrowPath className="animate-spin text-base" /> : <HiArrowUpTray className="text-base" />}
        {!collapsed && (uploading ? "Uploading..." : "Upload PDF")}
      </button>

      {!collapsed && (
        <p className="mt-2 text-xs text-slate-300">
          Drag and drop file here or click upload.
        </p>
      )}

      {uploading && !collapsed && (
        <div className="mt-3">
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-700/60">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${uploadProgress}%` }}
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-fuchsia-500"
            />
          </div>
          <p className="mt-1 text-right text-[11px] text-slate-400">{uploadProgress}%</p>
        </div>
      )}
    </motion.div>
  );
}

export default FileUpload;