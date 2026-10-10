"use client";

import { useEffect, useRef, useState } from "react";
import { CircleNotch, DownloadSimple, X } from "@phosphor-icons/react";
import type { ExportFormat } from "@/lib/types";
import { saveBlob } from "@/lib/api";
import { MarkdownLite } from "@/components/MarkdownLite";

/**
 * Full-screen preview of the exact file the export endpoint returned:
 * PDF in the browser's viewer, DOCX rendered client-side with docx-preview,
 * Markdown through the app's own renderer.
 */
export function ExportPreview({
  file,
  format,
  presetLabel,
  onClose,
}: {
  file: { blob: Blob; filename: string };
  format: ExportFormat;
  presetLabel?: string;
  onClose: () => void;
}) {
  const docxRef = useRef<HTMLDivElement>(null);
  // One preview per mount: the PDF URL is created once and revoked on close.
  const [pdfUrl] = useState(() =>
    format === "pdf" ? URL.createObjectURL(file.blob) : null,
  );
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [rendering, setRendering] = useState(format === "docx");

  useEffect(() => {
    if (pdfUrl) {
      return () => URL.revokeObjectURL(pdfUrl);
    }
    if (format === "md") {
      void file.blob.text().then(setMarkdown);
      return;
    }
    let cancelled = false;
    // Loaded on demand: the renderer is only needed when someone previews.
    void import("docx-preview").then(async ({ renderAsync }) => {
      if (cancelled || !docxRef.current) return;
      await renderAsync(file.blob, docxRef.current, undefined, {
        breakPages: true,
        ignoreLastRenderedPageBreak: true,
      });
      if (!cancelled) setRendering(false);
    });
    return () => {
      cancelled = true;
    };
  }, [file, format, pdfUrl]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-ink/40 p-3 backdrop-blur-sm sm:p-6"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Xem trước tài liệu"
        className="animate-enter mx-auto flex min-h-0 w-full max-w-5xl flex-1 flex-col overflow-hidden rounded-[var(--radius-bezel)] bg-surface-2 shadow-[var(--shadow-lift)] ring-1 ring-line"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 border-b border-line bg-surface px-4 py-3 sm:px-5">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-ink">{file.filename}</p>
            <p className="text-xs text-ink-faint">
              Xem trước{presetLabel ? ` · mẫu ${presetLabel}` : ""}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => saveBlob(file.blob, file.filename)}
              className="btn btn-primary px-4 py-2"
            >
              <DownloadSimple size={16} weight="bold" />
              Tải xuống
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Đóng xem trước"
              className="flex h-9 w-9 items-center justify-center rounded-full text-ink-soft transition hover:bg-surface-3 hover:text-ink"
            >
              <X size={18} />
            </button>
          </div>
        </header>

        <div className="relative min-h-0 flex-1 overflow-auto">
          {format === "pdf" && pdfUrl && (
            <iframe title="Xem trước PDF" src={pdfUrl} className="h-full w-full" />
          )}
          {format === "md" && (
            <article className="mx-auto max-w-3xl bg-surface px-8 py-10 shadow-[var(--shadow-card)] sm:my-6 sm:rounded-2xl">
              {markdown !== null && <MarkdownLite text={markdown} />}
            </article>
          )}
          {format === "docx" && (
            <>
              {rendering && (
                <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-ink-faint">
                  <CircleNotch size={18} className="animate-spin" />
                  Đang dựng bản xem trước…
                </div>
              )}
              <div ref={docxRef} className="docx-preview-host" />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
