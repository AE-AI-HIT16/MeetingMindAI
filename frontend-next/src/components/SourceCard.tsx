"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useSession } from "next-auth/react";
import type { Source } from "@/lib/types";
import { formatDate, formatDuration } from "@/lib/format";
import { StatusBadge, Waveform } from "@/components/ui";
import { deleteSource } from "@/lib/api";

const DOC_LABEL: Record<string, string> = {
  live: "Đang tạo",
  summary: "Tóm tắt",
  full_text: "Toàn văn",
};

export function SourceCard({ source }: { source: Source }) {
  const router = useRouter();
  const { data: session } = useSession();
  const [deleting, setDeleting] = useState(false);

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
      alert("Không thể xóa tài liệu. Vui lòng thử lại.");
      setDeleting(false);
    }
  }

  return (
    <div className="group relative">
      <Link
        href={href}
        className="flex flex-col rounded-[var(--radius-card)] border border-line bg-surface p-5 shadow-[var(--shadow-card)] transition duration-300 hover:-translate-y-1 hover:border-transparent hover:shadow-[var(--shadow-lift)]"
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

      {/* Delete button — hiện khi hover */}
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
