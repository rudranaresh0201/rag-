import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";

function splitAnswerSections(answer) {
  const lines = String(answer || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const bulletLines = lines.filter((line) => /^[-*]|^\d+\./.test(line));
  const plainLines = lines.filter((line) => !/^[-*]|^\d+\./.test(line));

  const summary = plainLines.slice(0, 2).join(" ") || "No summary available.";

  const explanation = plainLines.slice(2).join(" ") || "No detailed explanation available.";

  return {
    summary,
    keyPoints: bulletLines.slice(0, 5),
    explanation,
  };
}

function useTypewriter(text, speed = 8) {
  const [rendered, setRendered] = useState("");

  useEffect(() => {
    if (!text) {
      setRendered("");
      return;
    }

    let index = 0;
    setRendered("");

    const timer = window.setInterval(() => {
      index += 1;
      setRendered(text.slice(0, index));
      if (index >= text.length) {
        window.clearInterval(timer);
      }
    }, speed);

    return () => window.clearInterval(timer);
  }, [text, speed]);

  return rendered;
}

function highlightKeywords(text, keywords) {
  if (!text) {
    return null;
  }

  const tokens = text.split(/(\s+)/);
  return tokens.map((token, index) => {
    const normalized = token.toLowerCase().replace(/[^a-z0-9]/g, "");
    const isMatch = keywords.includes(normalized) && normalized.length > 2;

    if (!isMatch) {
      return <span key={`${token}-${index}`}>{token}</span>;
    }

    return (
      <mark
        key={`${token}-${index}`}
        className="rounded-md bg-fuchsia-400/20 px-1 text-fuchsia-100"
      >
        {token}
      </mark>
    );
  });
}

function AnswerPanel({ answer, query, loading, sources = [] }) {
  const sections = useMemo(() => splitAnswerSections(answer), [answer]);

  const keywords = useMemo(
    () =>
      String(query || "")
        .toLowerCase()
        .split(/\s+/)
        .map((word) => word.trim().replace(/[^a-z0-9]/g, ""))
        .filter(Boolean),
    [query]
  );

  const typedSummary = useTypewriter(sections.summary, 10);
  const typedExplanation = useTypewriter(sections.explanation, 7);

  if (!answer && !loading) {
    return (
      <div className="rounded-3xl border border-white/15 bg-white/5 p-6 text-sm text-slate-300 backdrop-blur-xl">
        Ask a question to generate a structured answer.
      </div>
    );
  }

  return (
    <motion.div layout className="space-y-4">
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="rounded-2xl border border-indigo-300/30 bg-indigo-400/10 px-4 py-3 text-sm text-indigo-100"
          >
            Streaming response...
          </motion.div>
        )}
      </AnimatePresence>

      <section className="rounded-3xl border border-white/15 bg-white/5 p-5 backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Summary</p>
        <p className="mt-2 text-base leading-relaxed text-slate-100">
          {highlightKeywords(typedSummary, keywords)}
          {typedSummary.length < sections.summary.length && <span className="type-caret" />}
        </p>
      </section>

      <section className="rounded-3xl border border-white/15 bg-white/5 p-5 backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Key Points</p>
        <ul className="mt-3 space-y-2 text-sm text-slate-100">
          {sections.keyPoints.length ? (
            sections.keyPoints.map((point, index) => (
              <li
                key={`${point}-${index}`}
                className="rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-2"
              >
                {highlightKeywords(point, keywords)}
              </li>
            ))
          ) : (
            <li className="text-slate-300">No explicit bullet points detected in this answer.</li>
          )}
        </ul>
      </section>

      <section className="rounded-3xl border border-white/15 bg-white/5 p-5 backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Explanation</p>
        <p className="mt-2 text-sm leading-relaxed text-slate-200">
          {highlightKeywords(typedExplanation, keywords)}
          {typedExplanation.length < sections.explanation.length && <span className="type-caret" />}
        </p>
      </section>

      <section className="rounded-3xl border border-white/15 bg-white/5 p-5 backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Sources</p>
        {sources.length ? (
          <div className="mt-3 space-y-2">
            {sources.map((source, index) => (
              <div
                key={`${source.document}-${index}`}
                className="rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-2"
              >
                <p className="text-xs text-slate-300">
                  {source.document}
                  {Number.isFinite(source.page) ? ` · Page ${source.page}` : ""}
                </p>
                <p className="mt-1 text-sm text-slate-100">
                  {source.text}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-sm text-slate-300">No sources available.</p>
        )}
      </section>
    </motion.div>
  );
}

export default AnswerPanel;
