"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import type { Source } from "@/lib/types";
import { formatDate, formatDuration } from "@/lib/format";
import { DocModeBadge, StatusBadge, Waveform } from "@/components/ui";
import { deleteSource } from "@/lib/api";
import { ErrorNotice } from "@/components/ErrorNotice";
import {
  Check,
  CircleNotch,
  Trash,
  VideoCamera,
  Waveform as AudioIcon,
} from "@phosphor-icons/react";

export function SourceCard({
  source,
  selectionMode = false,
  selected = false,
  onToggleSelect,
  featured = false,
}: {
  source: Source;
  /** Wide variant for the first card of the library grid. */
  featured?: boolean;
  /** While selecting, a click toggles the card instead of opening it. */
  selectionMode?: boolean;
  selected?: boolean;
  onToggleSelect?: () => void;
}) {
  const router = useRouter();
  const { data: session } = useSession();
  const [deleting, setDeleting] = useState(false);
  // First click arms the delete button, a second click deletes.
  const [armed, setArmed] = useState(false);
  const [deleteError, setDeleteError] = useState<unknown>(null);

  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(false), 3000);
    return () => window.clearTimeout(timer);
  }, [armed]);

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
    if (!armed) {
      setArmed(true);
      return;
    }
    setArmed(false);
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
    <div className="group relative flex h-full flex-col">
      <Link
        href={href}
        onClick={(e) => {
          if (selectionMode) {
            e.preventDefault();
            onToggleSelect?.();
          }
        }}
        aria-pressed={selectionMode ? selected : undefined}
        className={`flex flex-1 flex-col rounded-[var(--radius-card)] border bg-surface p-5 shadow-[var(--shadow-card)] transition duration-300 hover:-translate-y-1 hover:shadow-[var(--shadow-lift)] ${
          selected
            ? "border-brand ring-2 ring-brand/40"
            : "border-line hover:border-ink-faint/40"
        }`}
      >
        {/* Media preview strip */}
        <div
          className={`relative mb-4 flex items-end overflow-hidden rounded-xl bg-surface-2 p-3 ring-1 ring-line-soft ${
            featured ? "h-36" : "h-28"
          }`}
        >
          <Waveform
            live={source.status === "processing"}
            bars={featured ? 72 : 32}
            className="absolute inset-x-4 inset-y-6 opacity-60 transition-opacity duration-300 group-hover:opacity-90"
          />
          <span className="relative flex items-center gap-1.5 rounded-md bg-paper/80 px-2 py-1 text-[11px] font-semibold text-ink-soft ring-1 ring-line backdrop-blur">
            {isVideo ? <VideoCamera size={13} /> : <AudioIcon size={13} />}
            {isVideo ? "Video" : "Audio"}
          </span>
        </div>

        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-xs text-ink-faint">
            {formatDuration(source.durationMs)}
          </span>
          <StatusBadge status={source.status} />
        </div>

        <h3
          className={`mt-2 line-clamp-2 font-display font-semibold leading-snug tracking-tight text-ink [overflow-wrap:anywhere] transition group-hover:text-brand-ink ${
            featured ? "text-2xl" : "text-[17px]"
          }`}
        >
          {source.title}
        </h3>

        <div className="mt-auto flex items-center justify-between gap-2 pt-4">
          <span className="text-xs text-ink-faint">
            {formatDate(source.createdAt)}
          </span>
          <div className="flex gap-1.5">
            {source.docs.map((d) => (
              <DocModeBadge key={d} mode={d} />
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
              ? "bg-brand text-on-brand ring-brand opacity-100"
              : `bg-surface-2 text-transparent ring-line hover:ring-brand ${
                  selectionMode
                    ? "opacity-100"
                    : "opacity-0 group-hover:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
                }`
          }`}
        >
          <Check size={14} weight="bold" aria-hidden="true" />
        </button>
      )}

      {/* Delete button — hiện khi hover */}
      {!selectionMode && (
      <button
        onClick={handleDelete}
        onBlur={() => setArmed(false)}
        disabled={deleting}
        title={armed ? "Bấm lần nữa để xóa" : "Xóa tài liệu"}
        aria-label={armed ? `Xác nhận xóa "${source.title}"` : `Xóa "${source.title}"`}
        className={`absolute right-2 top-2 flex h-7 items-center justify-center gap-1 rounded-lg shadow-sm ring-1 transition group-hover:opacity-100 focus-visible:opacity-100 disabled:opacity-50 [@media(hover:none)]:opacity-100 ${
          armed
            ? "bg-danger px-2 text-xs font-medium text-on-brand opacity-100 ring-danger"
            : "w-7 bg-surface-2 text-ink-soft opacity-0 ring-line hover:bg-danger-wash hover:text-danger-ink hover:ring-danger/30"
        }`}
      >
        {armed ? (
          "Xóa?"
        ) : deleting ? (
          <CircleNotch size={13} className="animate-spin" />
        ) : (
          <Trash size={13} />
        )}
      </button>
      )}
    </div>
  );
}
