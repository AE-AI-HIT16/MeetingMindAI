"use client";

import { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownLiteProps {
  text: string;
  /** When true, highlights the last paragraph as it streams in */
  live?: boolean;
}

/** Render persisted/streamed Markdown as safe semantic HTML. */
export function MarkdownLite({ text, live = false }: MarkdownLiteProps) {
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
    <div ref={containerRef} className="markdown-preview">
      <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>
        {text}
      </ReactMarkdown>
    </div>
  );
}
