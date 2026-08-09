import Link from "next/link";
import type { Source } from "@/lib/types";
import { formatDate, formatDuration } from "@/lib/format";
import { StatusBadge, Waveform } from "@/components/ui";

const DOC_LABEL: Record<string, string> = {
  live: "Đang tạo",
  summary: "Tóm tắt",
  full_text: "Toàn văn",
};

export function SourceCard({ source }: { source: Source }) {
  const isVideo = source.mediaType === "video";
  const finalDocument =
    source.documents.find((document) => document.mode === "summary") ??
    source.documents.find((document) => document.mode === "full_text");
  const href = finalDocument
    ? `/sources/${source.id}?view=doc&documentId=${finalDocument.id}`
    : `/sources/${source.id}`;

  return (
    <Link
      href={href}
      className="group flex flex-col rounded-[var(--radius-card)] border border-line bg-surface p-5 shadow-[var(--shadow-card)] transition duration-300 hover:-translate-y-1 hover:border-transparent hover:shadow-[var(--shadow-lift)]"
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
