"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import { updateSegment } from "@/lib/api";

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
}: {
  jobId: string | null;
  segmentId: number | null;
  text: string;
  className?: string;
}) {
  const { data: authSession } = useSession();
  const [saved, setSaved] = useState<string | null>(null);
  const [draft, setDraft] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      setDraft(null);
    } catch {
      setError("Không thể lưu chỉnh sửa. Vui lòng thử lại.");
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
            className="rounded-md bg-brand px-3 py-1 text-xs font-medium text-white transition hover:bg-brand-ink disabled:opacity-50"
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
        {error && <p className="mt-1 text-xs text-danger">{error}</p>}
      </div>
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
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
          </svg>
          Sửa
        </button>
      )}
    </p>
  );
}
