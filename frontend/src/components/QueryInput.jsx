import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { HiOutlineSparkles } from "react-icons/hi2";

const PLACEHOLDERS = [
  "Ask across all uploaded documents...",
  "Summarize the key findings from my PDFs",
  "Compare definitions and cite exact evidence",
  "What are the main risks and constraints?",
];

function QueryInput({
  value,
  onChange,
  onSubmit,
  disabled,
  hero = false,
  loading = false,
  canSubmit = true,
}) {
  const [placeholderIndex, setPlaceholderIndex] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setPlaceholderIndex((current) => (current + 1) % PLACEHOLDERS.length);
    }, 2800);

    return () => window.clearInterval(timer);
  }, []);

  const activePlaceholder = useMemo(
    () => PLACEHOLDERS[placeholderIndex],
    [placeholderIndex]
  );

  const handleSend = () => {
    if (disabled || !canSubmit || !value.trim()) {
      return;
    }
    onSubmit();
  };

  return (
    <motion.div
      layout
      initial={hero ? { opacity: 0, y: 16 } : false}
      animate={{ opacity: 1, y: 0 }}
      className={
        hero
          ? "mx-auto w-full max-w-3xl"
          : "w-full rounded-3xl border border-white/15 bg-white/5 p-3 backdrop-blur-xl"
      }
    >
      <div className={hero ? "rounded-[2rem] border border-white/15 bg-white/5 p-3 backdrop-blur-xl" : ""}>
        <div className="flex items-end gap-3">
          <div className="relative flex-1">
            <textarea
              rows={hero ? 3 : 2}
              value={value}
              disabled={disabled}
              onChange={(event) => onChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSend();
                }
              }}
              placeholder=" "
              className="scrollbar-thin min-h-[64px] w-full resize-none rounded-2xl border border-white/15 bg-slate-950/65 px-4 py-3 pr-28 text-sm text-slate-100 outline-none transition focus:border-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
            />

            {!value && (
              <div className="pointer-events-none absolute left-4 top-3 text-sm text-slate-400">
                <AnimatePresence mode="wait">
                  <motion.span
                    key={activePlaceholder}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.35 }}
                    className="inline-flex items-center gap-2"
                  >
                    <HiOutlineSparkles className="text-indigo-300" />
                    {activePlaceholder}
                    <span className="type-caret" />
                  </motion.span>
                </AnimatePresence>
              </div>
            )}
          </div>

          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            type="button"
            onClick={handleSend}
            disabled={disabled || !canSubmit || !value.trim()}
            className="h-12 rounded-2xl bg-gradient-to-r from-indigo-500 via-blue-500 to-fuchsia-500 px-5 text-sm font-semibold text-white shadow-lg shadow-indigo-500/25 transition disabled:cursor-not-allowed disabled:opacity-45"
          >
            {loading ? "Thinking..." : "Ask"}
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}

export default QueryInput;
