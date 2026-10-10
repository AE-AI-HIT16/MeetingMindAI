"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import { updateSegment } from "@/lib/api";
import { ErrorNotice } from "@/components/ErrorNotice";
import { PencilSimple } from "@phosphor-icons/react";

/**
 * One transcript sentence with an always-visible "Sửa" button (also
 * double-click). Saves via PATCH /v1/jobs/{jobId}/segments/{segmentId}.
 * Without a persisted id (jobId/segmentId null) it is plain text.
 */
export function EditableSegmentText({
  jobId,
  segmentId,
  text,
  className = "",
  inline = false,
  onClick,
  onSaved,
}: {
  jobId: string | null;
  segmentId: number | null;
  text: string;
  className?: string;
  /** Render as a sentence inside a paragraph (speaker block). */
  inline?: boolean;
  /** Single click on the text (e.g. play from this sentence). */
  onClick?: () => void;
  /** Called with the saved text so the parent can keep it across remounts. */
  onSaved?: (text: string) => void;
}) {
  const { data: authSession } = useSession();
  const [saved, setSaved] = useState<string | null>(null);
  const [draft, setDraft] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const shown = saved ?? text;
  const editable = jobId !== null && segmentId !== null;

  const startEdit = () => {
    if (!editable) return;
    setDraft(shown);
    setError(null);
  };

  async function save() {
    if (jobId === null || segmentId === null || draft === null) return;
    if (draft.trim() === shown.trim()) {
      setDraft(null);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const updated = await updateSegment(jobId, segmentId, { text: draft }, authSession?.accessToken);
      setSaved(updated.text);
      onSaved?.(updated.text);
      setDraft(null);
    } catch (err) {
      setError(err);
    } finally {
      setSaving(false);
    }
  }

  if (draft !== null) {
    return (
      <div className="mt-1">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void save();
            if (e.key === "Escape") setDraft(null);
          }}
          rows={2}
          autoFocus
          className="w-full resize-y rounded-lg border border-brand/40 bg-surface px-2.5 py-1.5 text-[15px] leading-relaxed text-ink focus:border-brand focus:outline-none"
        />
        <div className="mt-1.5 flex items-center gap-2">
          <button
            type="button"
            onClick={() => void save()}
            disabled={saving}
            className="rounded-md bg-brand px-3 py-1 text-xs font-medium text-on-brand transition hover:bg-brand-ink disabled:opacity-50"
          >
            {saving ? "Đang lưu…" : "Lưu"}
          </button>
          <button
            type="button"
            onClick={() => setDraft(null)}
            className="rounded-md border border-line px-3 py-1 text-xs text-ink-soft transition hover:bg-surface-2"
          >
            Hủy
          </button>
          <span className="text-[11px] text-ink-faint">Ctrl+Enter để lưu · Esc để hủy</span>
        </div>
        {error ? (
          <ErrorNotice
            compact
            error={error}
            title="Chưa lưu được chỉnh sửa"
            onRetry={() => void save()}
            className="mt-2"
          />
        ) : null}
      </div>
    );
  }

  if (inline) {
    // Inside a speaker block: the block header has the visible "Sửa" button.
    return (
      <span
        className={`cursor-pointer rounded-sm leading-relaxed text-ink ${className}`}
        onClick={onClick}
        onDoubleClick={startEdit}
        title={editable ? "Bấm để nghe · nhấp đúp để sửa" : "Bấm để nghe"}
      >
        {shown}
      </span>
    );
  }

  return (
    <p
      className={`leading-relaxed text-ink ${className}`}
      onDoubleClick={startEdit}
      title={editable ? "Nhấp đúp để sửa câu này" : undefined}
    >
      {shown}
      {editable && (
        <button
          type="button"
          onClick={startEdit}
          aria-label="Sửa câu này"
          className="ml-2 inline-flex translate-y-[-1px] items-center gap-1 rounded-md border border-line px-1.5 py-0.5 align-middle text-[11px] font-medium text-ink-soft transition hover:border-brand hover:bg-brand-wash hover:text-brand-ink"
        >
          <PencilSimple size={11} aria-hidden="true" />
          Sửa
        </button>
      )}
    </p>
  );
}
