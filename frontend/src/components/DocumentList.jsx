import { motion } from "framer-motion";
import { HiTrash, HiDocumentText } from "react-icons/hi2";

function DocumentList({
  uploadedFiles,
  activeDocumentId,
  onSelectDocument,
  onDeleteDocument,
  collapsed = false,
}) {
  const documents = Array.isArray(uploadedFiles) ? uploadedFiles : [];

  return (
    <div className="space-y-2">
      {documents?.length > 0 ? (
        documents.map((file) => {
        const doc = file || {};
        console.log("DOCUMENT ITEM:", doc);
        const isActive = activeDocumentId === doc.id;

        return (
          <motion.div
            key={doc.id}
            whileHover={{ y: -1 }}
            className={`rounded-2xl border p-2 transition ${
              isActive
                ? "border-indigo-300/60 bg-indigo-400/12"
                : "border-white/15 bg-slate-900/35 hover:border-indigo-300/35"
            }`}
          >
            <button
              type="button"
              onClick={() => onSelectDocument(doc.id)}
              className="w-full text-left"
              title={doc.name}
            >
              <div className="inline-flex items-center gap-2">
                <HiDocumentText className="text-sm text-indigo-200" />
                {!collapsed && (
                  <span className="line-clamp-1 text-sm font-semibold text-slate-100">{doc.name}</span>
                )}
              </div>
              {!collapsed && (
                <p className="mt-1 text-[11px] text-slate-400">{doc.chunk_count || doc.chunks || "Available"} chunks</p>
              )}
            </button>

            {!collapsed && (
              <div className="mt-2 flex items-center justify-between">
                <span className="text-[11px] text-slate-300">{isActive ? "Active" : "Available"}</span>
                <button
                  type="button"
                  onClick={() => {}}
                  disabled
                  className="inline-flex items-center gap-1 rounded-lg border border-rose-300/20 px-2 py-1 text-[11px] font-semibold text-rose-200/60"
                >
                  <HiTrash />
                  Delete
                </button>
              </div>
            )}
          </motion.div>
        );
      })
      ) : (
        <p className="rounded-2xl border border-white/15 bg-slate-900/45 p-3 text-xs text-slate-300">
          {collapsed ? "0" : "No documents"}
        </p>
      )}
    </div>
  );
}

export default DocumentList;