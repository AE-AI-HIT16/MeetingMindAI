"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { useSession } from "next-auth/react";
import type { Source } from "@/lib/types";
import { deleteSource } from "@/lib/api";
import { computeLibraryStats, formatTotalDuration } from "@/lib/libraryStats";
import { SourceCard } from "@/components/SourceCard";
import { ErrorNotice } from "@/components/ErrorNotice";

const DELETE_CONCURRENCY = 4;

/** Library overview, the grid of sources and multi-select bulk delete. */
export function LibraryGrid({ sources }: { sources: Source[] }) {
  const router = useRouter();
  const { data: session } = useSession();
  const stats = useMemo(() => computeLibraryStats(sources), [sources]);

  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirming, setConfirming] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [failed, setFailed] = useState<string[]>([]);

  const selectionMode = selecting || selected.size > 0;
  const allSelected = sources.length > 0 && selected.size === sources.length;

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setConfirming(false);
  }

  function exitSelection() {
    setSelecting(false);
    setSelected(new Set());
    setConfirming(false);
  }

  async function deleteSelected() {
    const ids = [...selected];
    setConfirming(false);
    setFailed([]);
    setProgress({ done: 0, total: ids.length });
    const failures: string[] = [];
    let next = 0;
    let done = 0;
    async function worker() {
      while (next < ids.length) {
        const id = ids[next++];
        try {
          await deleteSource(id, session?.accessToken);
        } catch (err) {
          console.error("[LibraryGrid] delete failed:", id, err);
          failures.push(id);
        }
        done += 1;
        setProgress({ done, total: ids.length });
      }
    }
    await Promise.all(
      Array.from({ length: Math.min(DELETE_CONCURRENCY, ids.length) }, worker),
    );
    setProgress(null);
    setFailed(failures);
    setSelected(new Set(failures));
    if (failures.length === 0) setSelecting(false);
    router.refresh();
  }

  const failedTitles = sources
    .filter((source) => failed.includes(source.id))
    .map((source) => source.title);

  return (
    <>
      {/* Overview */}
      <dl className="mt-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile
          label="Tổng tài liệu"
          value={String(stats.total)}
          detail={`${stats.audio} audio · ${stats.video} video`}
        />
        <StatTile
          label="Tổng thời lượng"
          value={formatTotalDuration(stats.totalDurationMs)}
          detail="Audio và video đã tải lên"
        />
        <StatTile
          label="Đã có tài liệu"
          value={`${stats.withDocument}/${stats.total}`}
          detail={`${stats.summaries} tóm tắt · ${stats.fullTexts} toàn văn`}
        />
        <StatTile
          label="Trạng thái"
          value={stats.processing > 0 ? `${stats.processing} đang xử lý` : "Đã xử lý xong"}
          detail={stats.failed > 0 ? `${stats.failed} bị lỗi` : "Không có lỗi"}
        />
      </dl>

      {/* Selection toolbar */}
      {sources.length > 0 && (
        <div className="sticky top-0 z-10 mt-6 flex flex-wrap items-center gap-2 rounded-xl border border-line bg-surface/95 px-3 py-2 backdrop-blur">
          {!selectionMode ? (
            <>
              <span className="text-sm text-ink-soft">{stats.total} tài liệu</span>
              <button
                type="button"
                onClick={() => setSelecting(true)}
                className="ml-auto rounded-lg border border-line px-3 py-1.5 text-sm font-medium text-ink-soft transition hover:border-brand hover:text-brand-ink"
              >
                Chọn nhiều
              </button>
            </>
          ) : (
            <>
              <span className="text-sm font-medium text-ink">
                Đã chọn {selected.size}/{sources.length}
              </span>
              <button
                type="button"
                onClick={() =>
                  setSelected(allSelected ? new Set() : new Set(sources.map((s) => s.id)))
                }
                disabled={progress !== null}
                className="rounded-lg border border-line px-3 py-1.5 text-sm text-ink-soft transition hover:border-brand hover:text-brand-ink disabled:opacity-50"
              >
                {allSelected ? "Bỏ chọn tất cả" : "Chọn tất cả"}
              </button>
              <div className="ml-auto flex items-center gap-2">
                {progress ? (
                  <span className="text-sm text-ink-soft" role="status">
                    Đang xóa {progress.done}/{progress.total}…
                  </span>
                ) : confirming ? (
                  <>
                    <span className="text-sm text-ink">
                      Xóa vĩnh viễn {selected.size} tài liệu (cả media và tài liệu đã tạo)?
                    </span>
                    <button
                      type="button"
                      onClick={() => setConfirming(false)}
                      className="rounded-lg border border-line px-3 py-1.5 text-sm text-ink-soft transition hover:bg-surface-2"
                    >
                      Hủy
                    </button>
                    <button
                      type="button"
                      onClick={() => void deleteSelected()}
                      className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-red-700"
                    >
                      Xóa {selected.size} tài liệu
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={() => setConfirming(true)}
                      disabled={selected.size === 0}
                      className="rounded-lg border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Xóa{selected.size > 0 ? ` (${selected.size})` : ""}
                    </button>
                    <button
                      type="button"
                      onClick={exitSelection}
                      className="rounded-lg border border-line px-3 py-1.5 text-sm text-ink-soft transition hover:bg-surface-2"
                    >
                      Xong
                    </button>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {failedTitles.length > 0 && (
        <ErrorNotice
          error={`Chưa xóa được ${failedTitles.length} tài liệu: ${failedTitles.slice(0, 3).join(", ")}${
            failedTitles.length > 3 ? "…" : ""
          }. Các tài liệu này vẫn đang được chọn để bạn thử lại.`}
          title="Một số tài liệu chưa xóa được"
          onDismiss={() => setFailed([])}
          className="mt-4"
        />
      )}

      <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <Link
          href="/upload"
          className="group flex min-h-[260px] flex-col items-center justify-center rounded-[var(--radius-card)] border-2 border-dashed border-line bg-surface/50 p-5 text-center transition hover:border-brand hover:bg-brand-wash/40"
        >
          <span className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-wash text-brand transition group-hover:scale-110">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </span>
          <span className="mt-3 font-display text-lg font-medium text-ink">
            Tài liệu mới
          </span>
          <span className="mt-1 text-xs text-ink-faint">
            Kéo thả hoặc chọn tệp media
          </span>
        </Link>

        {sources.map((s) => (
          <SourceCard
            key={s.id}
            source={s}
            selectionMode={selectionMode}
            selected={selected.has(s.id)}
            onToggleSelect={() => toggle(s.id)}
          />
        ))}
      </div>
    </>
  );
}

function StatTile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3">
      <dt className="text-xs font-medium text-ink-faint">{label}</dt>
      <dd className="mt-1 font-display text-2xl font-medium tabular-nums text-ink">{value}</dd>
      <dd className="mt-0.5 text-xs text-ink-soft">{detail}</dd>
    </div>
  );
}
