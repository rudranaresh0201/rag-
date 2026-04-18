import { AnimatePresence, motion } from "framer-motion";
import { HiBars3 } from "react-icons/hi2";
import QueryInput from "./QueryInput";

function ChatLayout({
  started,
  query,
  onChangeQuery,
  onSubmit,
  disabled,
  loading,
  canSubmit,
  sidebar,
  sidebarCollapsed,
  mobileSidebarOpen,
  onOpenMobileSidebar,
  onCloseMobileSidebar,
  history,
  rightPanel,
}) {
  const gridColumns = sidebarCollapsed
    ? "lg:grid-cols-[86px_1fr_480px]"
    : "lg:grid-cols-[300px_1fr_480px]";

  return (
    <div className={`relative grid min-h-[90vh] w-full grid-cols-1 gap-4 ${gridColumns}`}>
      <button
        type="button"
        onClick={onOpenMobileSidebar}
        className="fixed left-4 top-4 z-30 inline-flex h-10 w-10 items-center justify-center rounded-xl border border-white/15 bg-slate-900/75 text-slate-100 backdrop-blur-xl transition hover:border-indigo-300/50 lg:hidden"
      >
        <HiBars3 />
      </button>

      <AnimatePresence>
        {mobileSidebarOpen && (
          <>
            <motion.button
              type="button"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onCloseMobileSidebar}
              className="fixed inset-0 z-30 bg-slate-950/55 lg:hidden"
            />
            <motion.aside
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: "spring", stiffness: 220, damping: 28 }}
              className="fixed left-0 top-0 z-40 h-full w-[300px] border-r border-white/15 bg-slate-950/95 p-4 backdrop-blur-xl lg:hidden"
            >
              {sidebar}
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <aside className="hidden rounded-3xl border border-white/15 bg-white/5 p-4 backdrop-blur-xl lg:block">
        {sidebar}
      </aside>

      <section className="flex min-h-0 flex-col gap-4">
        <div className="min-h-0 flex-1 rounded-3xl border border-white/15 bg-white/5 p-4 backdrop-blur-xl">
          <div className="scrollbar-thin h-full overflow-y-auto pr-2">
            {started ? (
              history
            ) : (
              <div className="mx-auto flex h-full min-h-[66vh] w-full max-w-4xl flex-col">
                <div className="flex-1" />
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-8"
                >
                  <div className="space-y-3 text-center">
                    <p className="inline-block rounded-full border border-indigo-300/30 bg-indigo-400/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-indigo-200">
                      AI Research Workspace
                    </p>
                    <h1 className="hero-title text-4xl font-semibold text-slate-100 md:text-6xl">
                      Ask With Evidence,
                      <span className="block bg-gradient-to-r from-blue-300 via-indigo-300 to-fuchsia-300 bg-clip-text text-transparent">
                        Think With Structure
                      </span>
                    </h1>
                    <p className="mx-auto max-w-2xl text-sm text-slate-300 md:text-base">
                      Notebook-style reasoning, Perplexity-like source grounding, and smooth interactions for your document intelligence flow.
                    </p>
                  </div>
                </motion.div>
                <div className="flex-1" />
              </div>
            )}
          </div>
        </div>

        <QueryInput
          value={query}
          onChange={onChangeQuery}
          onSubmit={onSubmit}
          disabled={disabled}
          loading={loading}
          canSubmit={canSubmit}
        />
      </section>

      <AnimatePresence mode="wait">
        <motion.section
          key="right-panel-pane"
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 12 }}
          className="rounded-3xl border border-white/15 bg-white/5 p-4 backdrop-blur-xl"
        >
          {rightPanel}
        </motion.section>
      </AnimatePresence>
    </div>
  );
}

export default ChatLayout;
