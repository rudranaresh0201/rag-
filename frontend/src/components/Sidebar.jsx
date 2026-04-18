import { motion } from "framer-motion";
import UploadBox from "./UploadBox";

const MotionDiv = motion.div;

function formatSize(size) {
  if (!size && size !== 0) {
    return "Unknown size";
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function Sidebar({
  uploadedFiles,
  activeDocumentId,
  uploading,
  uploadProgress,
  onUpload,
  onSelectDocument,
  onDeleteFile,
  onClearAll,
  clearing,
}) {
  const navItems = [
    { label: "Documents", active: true },
    { label: "Collections", active: false },
    { label: "Chat History", active: false },
  ];

  return (
    <aside className="flex h-full flex-col rounded-3xl border border-slate-200 bg-slate-50 p-4 shadow-md md:p-5">
      <h2 className="text-xl font-semibold tracking-tight text-slate-900">Workspace</h2>
      <p className="mt-1 text-sm text-slate-500">Your AI document workspace.</p>

      <div className="mt-5">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Workspace</p>
        <div className="space-y-1.5">
          {navItems.map((item) => (
            <div
              key={item.label}
              className={`rounded-xl px-3 py-2 text-sm transition ${
                item.active
                  ? "bg-blue-50 text-blue-700 ring-1 ring-blue-100"
                  : "text-slate-500 hover:bg-slate-100 hover:text-slate-700"
              }`}
            >
              {item.label}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-5">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Actions</p>
        <UploadBox onUpload={onUpload} uploading={uploading} uploadProgress={uploadProgress} />
      </div>

      <div className="scrollbar-thin mt-5 flex-1 space-y-2 overflow-y-auto pr-1">
        {uploadedFiles.length === 0 && (
          <div className="rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-500">
            No PDFs uploaded yet.
          </div>
        )}

        {uploadedFiles.map((file) => (
          <MotionDiv
            key={file.id}
            whileHover={{ y: -2 }}
            className={`group rounded-xl border bg-white p-3 transition hover:border-blue-200 hover:shadow-md ${
              activeDocumentId === file.id
                ? "border-blue-300 ring-2 ring-blue-100"
                : "border-slate-200"
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <button
                type="button"
                onClick={() => onSelectDocument(file.id)}
                className="min-w-0 flex-1 text-left"
              >
                <p className="line-clamp-1 text-sm font-medium text-slate-800">{file.name}</p>
                <p className="mt-1 text-xs text-slate-500">{formatSize(file.size)}</p>
              </button>
              <div className="flex items-center gap-1">
                {activeDocumentId === file.id && (
                  <span className="h-2 w-2 rounded-full bg-blue-500" />
                )}
                <button
                  type="button"
                  onClick={() => onDeleteFile(file.id)}
                  className="rounded-lg border border-slate-300 px-2 py-1 text-xs text-slate-600 transition hover:border-red-200 hover:bg-red-50 hover:text-red-500"
                >
                  Delete
                </button>
              </div>
            </div>
          </MotionDiv>
        ))}
      </div>

      <button
        type="button"
        onClick={() => onSelectDocument(null)}
        className="mb-2 mt-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-50"
      >
        Search Across All Documents
      </button>

      <button
        type="button"
        onClick={onClearAll}
        disabled={clearing || uploading || uploadedFiles.length === 0}
        className="mt-2 rounded-xl border border-violet-200 bg-white px-3 py-2 text-sm font-medium text-violet-600 transition hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-45"
      >
        {clearing ? "Clearing..." : "Clear All"}
      </button>
    </aside>
  );
}

export default Sidebar;
