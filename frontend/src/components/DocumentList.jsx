import { motion } from "framer-motion";
import { HiTrash, HiDocumentText } from "react-icons/hi2";

function DocumentList({
  uploadedFiles,
  activeDocumentId,
  onSelectDocument,
  onDeleteDocument,
  collapsed = false,
}) {
  if (!uploadedFiles.length) {
    return (
      <div className="rounded-2xl border border-white/15 bg-slate-900/45 p-3 text-xs text-slate-300">
        {collapsed ? "0" : "No documents uploaded yet."}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {uploadedFiles.map((file) => {
        const isActive = activeDocumentId === file.id;

        return (
          <motion.div
            key={file.id}
            whileHover={{ y: -1 }}
            className={`rounded-2xl border p-2 transition ${
              isActive
                ? "border-indigo-300/60 bg-indigo-400/12"
                : "border-white/15 bg-slate-900/35 hover:border-indigo-300/35"
            }`}
          >
            <button
              type="button"
              onClick={() => onSelectDocument(file.id)}
              className="w-full text-left"
              title={file.name}
            >
              <div className="inline-flex items-center gap-2">
                <HiDocumentText className="text-sm text-indigo-200" />
                {!collapsed && (
                  <span className="line-clamp-1 text-sm font-semibold text-slate-100">{file.name}</span>
                )}
              </div>
              {!collapsed && (
                <p className="mt-1 text-[11px] text-slate-400">{file.chunks || 0} chunks</p>
              )}
            </button>

            {!collapsed && (
              <div className="mt-2 flex items-center justify-between">
                <span className="text-[11px] text-slate-300">{isActive ? "Active" : "Available"}</span>
                <button
                  type="button"
                  onClick={() => onDeleteDocument(file.id)}
                  className="inline-flex items-center gap-1 rounded-lg border border-rose-300/40 px-2 py-1 text-[11px] font-semibold text-rose-200 transition hover:bg-rose-400/15"
                >
                  <HiTrash />
                  Delete
                </button>
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}

export default DocumentList;