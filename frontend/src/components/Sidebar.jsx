import { motion } from "framer-motion";
import {
  HiBars3BottomLeft,
  HiFolder,
  HiSquares2X2,
  HiChatBubbleOvalLeftEllipsis,
  HiMiniSparkles,
} from "react-icons/hi2";
import DocumentList from "./DocumentList";
import FileUpload from "./FileUpload";

const NAV_ITEMS = [
  { icon: HiFolder, label: "Documents", active: true },
  { icon: HiSquares2X2, label: "Collections", active: false },
  { icon: HiChatBubbleOvalLeftEllipsis, label: "Sessions", active: false },
];

function Sidebar({
  uploadedFiles,
  activeDocumentId,
  uploading,
  processingDoc,
  uploadProgress,
  onUpload,
  onSelectDocument,
  onDeleteFile,
  onClearAll,
  clearing,
  collapsed = false,
  onToggleCollapsed,
}) {
  return (
    <div className="flex h-full min-h-[600px] flex-col gap-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="inline-flex items-center gap-1 rounded-full border border-indigo-300/30 bg-indigo-500/12 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-indigo-100">
            <HiMiniSparkles />
            {!collapsed && "Workspace"}
          </p>
          {!collapsed && <h2 className="mt-2 text-lg font-semibold text-slate-100">RAG Sidebar</h2>}
        </div>

        <button
          type="button"
          onClick={onToggleCollapsed}
          className="rounded-xl border border-white/15 bg-slate-900/55 p-2 text-slate-200 transition hover:border-indigo-300/50 hover:text-indigo-200"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <motion.span animate={{ rotate: collapsed ? 180 : 0 }} className="block">
            <HiBars3BottomLeft />
          </motion.span>
        </button>
      </div>

      <nav className="space-y-1">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.label}
              type="button"
              className={`inline-flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm transition ${
                item.active
                  ? "border border-indigo-300/35 bg-indigo-500/15 text-indigo-100"
                  : "text-slate-300 hover:bg-white/10"
              }`}
            >
              <Icon className="text-base" />
              {!collapsed && item.label}
            </button>
          );
        })}
      </nav>

      <FileUpload
        onUpload={onUpload}
        uploading={uploading}
        processingDoc={processingDoc}
        uploadProgress={uploadProgress}
        collapsed={collapsed}
      />

      <div className="scrollbar-thin flex-1 overflow-y-auto pr-1">
        <DocumentList
          uploadedFiles={uploadedFiles}
          activeDocumentId={activeDocumentId}
          onSelectDocument={onSelectDocument}
          onDeleteDocument={onDeleteFile}
          collapsed={collapsed}
        />
      </div>

      <div className="space-y-2">
        <button
          type="button"
          onClick={() => onSelectDocument(null)}
          className="w-full rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-xs font-semibold uppercase tracking-[0.13em] text-slate-100 transition hover:border-indigo-300/50"
          title="Search across all documents"
        >
          {collapsed ? "All" : "Search All Docs"}
        </button>

        <button
          type="button"
          onClick={onClearAll}
          disabled={clearing || uploading || Boolean(processingDoc) || uploadedFiles.length === 0}
          className="w-full rounded-xl border border-fuchsia-300/35 bg-fuchsia-500/12 px-3 py-2 text-xs font-semibold uppercase tracking-[0.13em] text-fuchsia-100 transition hover:bg-fuchsia-500/20 disabled:cursor-not-allowed disabled:opacity-50"
          title="Clear all documents"
        >
          {clearing ? "Clearing..." : collapsed ? "Clear" : "Clear All"}
        </button>
      </div>
    </div>
  );
}

export default Sidebar;
