import React from "react";

// Minimal markdown for the base build: renders **bold** inline.
// Swap for react-markdown + remark-gfm when wiring the real backend
// (see meet_docs/docs/12_frontend_nextjs_plan.md §1).
export function MarkdownLite({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <p className="text-[15px] leading-[1.75] text-ink-soft">
      {parts.map((p, i) =>
        p.startsWith("**") && p.endsWith("**") ? (
          <strong key={i} className="font-semibold text-ink">
            {p.slice(2, -2)}
          </strong>
        ) : (
          <React.Fragment key={i}>{p}</React.Fragment>
        )
      )}
    </p>
  );
}
