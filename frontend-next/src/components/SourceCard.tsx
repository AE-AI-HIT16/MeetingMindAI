"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useSession } from "next-auth/react";
import type { Source } from "@/lib/types";
import { formatDate, formatDuration } from "@/lib/format";
import { StatusBadge, Waveform } from "@/components/ui";
import { deleteSource } from "@/lib/api";
import { ErrorNotice } from "@/components/ErrorNotice";

const DOC_LABEL: Record<string, string> = {
  live: "Đang tạo",
  summary: "Tóm tắt",
  full_text: "Toàn văn",
};

export function SourceCard({
  source,
  selectionMode = false,
  selected = false,
  onToggleSelect,
}: {
  source: Source;
  /** While selecting, a click toggles the card instead of opening it. */
  selectionMode?: boolean;
  selected?: boolean;
  onToggleSelect?: () => void;
}) {
  const router = useRouter();
  const { data: session } = useSession();
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<unknown>(null);

  const isVideo = source.mediaType === "video";
  const finalDocument =
    source.documents.find((document) => document.mode === "summary") ??
    source.documents.find((document) => document.mode === "full_text");
  const href = finalDocument
    ? `/sources/${source.id}?view=doc&documentId=${finalDocument.id}`
    : `/sources/${source.id}`;

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm(`Xóa "${source.title}"?`)) return;
    setDeleting(true);
    try {
      await deleteSource(source.id, session?.accessToken);
      router.refresh();
    } catch (err) {
      console.error("[SourceCard] delete failed:", err);
      setDeleteError(err);
      setDeleting(false);
    }
  }

  return (
    <div className="group relative">
      <Link
        href={href}
        onClick={(e) => {
          if (selectionMode) {
            e.preventDefault();
            onToggleSelect?.();
          }
        }}
        aria-pressed={selectionMode ? selected : undefined}
        className={`flex flex-col rounded-[var(--radius-card)] border bg-surface p-5 shadow-[var(--shadow-card)] transition duration-300 hover:-translate-y-1 hover:shadow-[var(--shadow-lift)] ${
          selected
            ? "border-brand ring-2 ring-brand/40"
            : "border-line hover:border-transparent"
        }`}
      >
        {/* Media preview strip */}
        <div className="relative mb-4 flex h-28 items-center justify-center overflow-hidden rounded-xl bg-surface-2">
          <Waveform
            live={source.status === "processing"}
            bars={32}
            className="absolute inset-x-5 inset-y-8 opacity-70"
          />
          <span className="relative flex items-center gap-1.5 rounded-full bg-surface/90 px-2.5 py-1 font-mono text-[11px] font-bold text-ink-soft ring-1 ring-line backdrop-blur">
            {isVideo ? <VideoIcon /> : <AudioIcon />}
            {isVideo ? "VIDEO" : "AUDIO"}
          </span>
        </div>

        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-xs text-ink-faint">
            {formatDuration(source.durationMs)}
          </span>
          <StatusBadge status={source.status} />
        </div>

        <h3 className="mt-2 line-clamp-2 font-display text-[17px] font-medium leading-snug text-ink transition group-hover:text-brand-ink">
          {source.title}
        </h3>

        <div className="mt-auto flex items-center justify-between gap-2 pt-4">
          <span className="text-xs text-ink-faint">
            {formatDate(source.createdAt)}
          </span>
          <div className="flex gap-1.5">
            {source.docs.map((d) => (
              <span
                key={d}
                className="rounded-md bg-brand-wash px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wide text-brand-ink"
              >
                {DOC_LABEL[d]}
              </span>
            ))}
          </div>
        </div>
      </Link>

      {deleteError ? (
        <ErrorNotice
          compact
          error={deleteError}
          title="Chưa xóa được tài liệu"
          onDismiss={() => setDeleteError(null)}
          className="mt-2"
        />
      ) : null}

      {onToggleSelect && (
        <button
          type="button"
          role="checkbox"
          aria-checked={selected}
          aria-label={`Chọn "${source.title}"`}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onToggleSelect();
          }}
          className={`absolute left-2 top-2 flex h-7 w-7 items-center justify-center rounded-lg shadow-sm ring-1 transition ${
            selected
              ? "bg-brand text-white ring-brand opacity-100"
              : `bg-surface text-transparent ring-line hover:ring-brand ${
                  selectionMode ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                }`
          }`}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M5 12l5 5L20 7" />
          </svg>
        </button>
      )}

      {/* Delete button — hiện khi hover */}
      {!selectionMode && (
      <button
        onClick={handleDelete}
        disabled={deleting}
        title="Xóa tài liệu"
        className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-lg bg-surface opacity-0 shadow-sm ring-1 ring-line transition hover:bg-red-50 hover:text-red-500 hover:ring-red-200 group-hover:opacity-100 disabled:opacity-50"
      >
        {deleting ? (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="animate-spin">
            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
          </svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6l-1 14H6L5 6" />
            <path d="M10 11v6M14 11v6" />
            <path d="M9 6V4h6v2" />
          </svg>
        )}
      </button>
      )}
    </div>
  );
}

function VideoIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="5" width="15" height="14" rx="2" />
      <path d="M17 9l5-3v12l-5-3" />
    </svg>
  );
}
function AudioIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3v18M8 7v10M4 10v4M16 6v12M20 9v6" />
    </svg>
  );
}
