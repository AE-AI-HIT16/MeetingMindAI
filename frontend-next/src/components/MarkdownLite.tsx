"use client";

import { Children, isValidElement, useRef, useEffect, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Clock } from "@phosphor-icons/react";
import { SpeakerChip } from "@/components/ui";

interface MarkdownLiteProps {
  text: string;
  /** When true, highlights the last paragraph as it streams in */
  live?: boolean;
  /** "document" = the editorial reading layout used on the result page. */
  variant?: "plain" | "document";
  /** Seek the media player; time ranges and turn stamps become clickable. */
  onSeek?: (ms: number) => void;
  /** Full text: dim every turn except this speaker's (null = unknown speaker). */
  focus?: { speaker: number | null } | null;
}

/** Render persisted/streamed Markdown as safe semantic HTML. */
export function MarkdownLite({
  text,
  live = false,
  variant = "plain",
  onSeek,
  focus = null,
}: MarkdownLiteProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const prevLenRef = useRef(text.length);

  // When new text arrives, briefly flash the last paragraph to show it changed
  useEffect(() => {
    if (!live || !containerRef.current) return;
    if (text.length <= prevLenRef.current) return;
    prevLenRef.current = text.length;

    const el = containerRef.current;
    const last = el.querySelector("p:last-child, li:last-child, h1:last-child, h2:last-child, h3:last-child");
    if (!last) return;

    (last as HTMLElement).style.animation = "none";
    (last as HTMLElement).offsetHeight; // reflow
    (last as HTMLElement).style.animation = "rise 0.35s cubic-bezier(0.22,1,0.36,1) both";
  }, [text, live]);

  return (
    <div
      ref={containerRef}
      className={variant === "document" ? "markdown-preview doc-reading" : "markdown-preview"}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={variant === "document" ? documentComponents(onSeek, focus) : undefined}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Document variant: recognises the two shapes the pipeline writes
 *   summary:   "- **Topic** *(04:42–08:16)*"
 *   full text: "**[00:02] Người nói 1:** words…"
 * and renders them as reading UI instead of raw bold/italic.
 * ------------------------------------------------------------------ */

// "04:42–08:16", "04:42"; several may be joined with ";" or ",".
const RANGE = /^(\d{1,2}):(\d{2})(?:\s*[–-]\s*\d{1,2}:\d{2})?$/;
// "[00:02] Người nói 1:" or "[00:30] Chưa xác định:" (speaker unknown).
const TURN = /^\[(\d{1,2}):(\d{2})\]\s*(?:(?:Người nói|Speaker)\s*(\d+)|Chưa xác định)\s*:?$/i;

/** Parses "(00:36–02:48; 07:24–07:42)" into its ranges, or null if any part is not a time. */
function parseRanges(text: string): { label: string; ms: number }[] | null {
  const parts = text.replace(/^\(|\)$/g, "").split(/\s*[;,]\s*/);
  const ranges = parts.map((part) => {
    const m = RANGE.exec(part.trim());
    return m ? { label: part.trim(), ms: (Number(m[1]) * 60 + Number(m[2])) * 1000 } : null;
  });
  return ranges.every(Boolean) ? (ranges as { label: string; ms: number }[]) : null;
}

function plainText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(plainText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return plainText(node.props.children);
  return "";
}

/** Stable anchor id for a heading (shared with the table of contents). */
export function headingId(text: string): string {
  return (
    "muc-" +
    text
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .replace(/đ/gi, "d")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "")
  );
}

function TimeChip({ label, ms, onSeek }: { label: string; ms: number; onSeek?: (ms: number) => void }) {
  const className =
    "mx-0.5 inline-flex translate-y-[-1px] items-center gap-1 rounded-full bg-surface-2 px-2 py-0.5 align-middle font-mono text-[11px] font-medium not-italic text-ink-soft ring-1 ring-line-soft";
  if (!onSeek) {
    return (
      <span className={className}>
        <Clock size={11} />
        {label}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={() => onSeek(ms)}
      title={`Nghe từ ${label.split(/[–-]/)[0]}`}
      className={`${className} transition hover:bg-brand-wash hover:text-brand-ink hover:ring-brand/30`}
    >
      <Clock size={11} />
      {label}
    </button>
  );
}

function documentComponents(
  onSeek?: (ms: number) => void,
  focus?: { speaker: number | null } | null,
): Components {
  const em: Components["em"] = ({ children }) => {
    const ranges = parseRanges(plainText(children).trim());
    if (ranges) {
      return (
        <>
          {ranges.map((r) => (
            <TimeChip key={r.label} label={r.label} ms={r.ms} onSeek={onSeek} />
          ))}
        </>
      );
    }
    return <em>{children}</em>;
  };

  return {
    // Section numbers come from a CSS counter (.doc-h2::before).
    h2({ children }) {
      return (
        <h2 id={headingId(plainText(children))} className="doc-h2">
          {children}
        </h2>
      );
    },
    em,
    p({ children }) {
      const parts = Children.toArray(children);
      const first = parts[0];
      if (isValidElement(first) && first.type === "strong") {
        const m = TURN.exec(plainText(first).trim());
        if (m) {
          const ms = (Number(m[1]) * 60 + Number(m[2])) * 1000;
          // "Người nói N" is 1-based; SpeakerChip takes the 0-based index.
          const speaker = m[3] ? Number(m[3]) - 1 : null;
          const dimmed = focus != null && focus.speaker !== speaker;
          return (
            <div className={`doc-turn ${dimmed ? "doc-turn-dim" : ""}`}>
              <TimeChip label={`${m[1]}:${m[2]}`} ms={ms} onSeek={onSeek} />
              <div className="min-w-0">
                <SpeakerChip speaker={speaker} />
                <p className="doc-turn-text">{parts.slice(1)}</p>
              </div>
            </div>
          );
        }
      }
      // "**[Section title]** *(00:00–00:36)*" right under a heading repeats
      // the heading; keep only the time range as a quiet section line.
      if (
        isValidElement(first) &&
        first.type === "strong" &&
        /^\[.+\]$/.test(plainText(first).trim()) &&
        parts.slice(1).every((part) => typeof part === "string" ? !part.trim() : isValidElement(part) && part.type === em)
      ) {
        return <p className="doc-section-meta">{parts.slice(1)}</p>;
      }
      // A paragraph that is only italics ("Nguồn: …") is document metadata.
      if (parts.length === 1 && isValidElement(first) && first.type === em) {
        return <p className="doc-meta">{plainText(first)}</p>;
      }
      return <p>{children}</p>;
    },
  };
}
