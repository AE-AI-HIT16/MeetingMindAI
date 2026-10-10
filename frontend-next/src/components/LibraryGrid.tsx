"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { useSession } from "next-auth/react";
import type { Source } from "@/lib/types";
import { deleteSource } from "@/lib/api";
import { computeLibraryStats, formatTotalDuration } from "@/lib/libraryStats";
import { SourceCard } from "@/components/SourceCard";
import { ErrorNotice } from "@/components/ErrorNotice";
import { Waveform } from "@/components/ui";

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
      {/* One instrument panel, columns split by hairlines (not four cards) */}
      <dl className="mt-4 grid grid-cols-2 overflow-hidden rounded-[var(--radius-bezel)] bg-surface shadow-[var(--shadow-card)] ring-1 ring-line/80 lg:grid-cols-4 [&>div]:border-line [&>div:nth-child(odd)]:border-r lg:[&>div]:border-r lg:[&>div:last-child]:border-r-0 [&>div:nth-child(-n+2)]:border-b lg:[&>div]:border-b-0">
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
        <div className="sticky top-3 z-10 mt-12 flex flex-wrap items-center gap-2 rounded-2xl bg-surface/85 px-3 py-2 shadow-[var(--shadow-card)] ring-1 ring-line/80 backdrop-blur-xl">
          {!selectionMode ? (
            <>
              <h2 className="px-1 font-display text-lg font-semibold tracking-tight text-ink">
                Gần đây
                <span className="ml-2 font-mono text-xs font-normal tabular-nums text-ink-faint">
                  {stats.total}
                </span>
              </h2>
              <button
                type="button"
                onClick={() => setSelecting(true)}
                className="ml-auto rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm font-medium text-ink-soft transition hover:border-brand hover:text-brand-ink"
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
                className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm text-ink-soft transition hover:border-brand hover:text-brand-ink disabled:opacity-50"
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
                      className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm text-ink-soft transition hover:bg-surface-2"
                    >
                      Hủy
                    </button>
                    <button
                      type="button"
                      onClick={() => void deleteSelected()}
                      className="rounded-full bg-danger px-3.5 py-1.5 text-sm font-medium text-on-brand transition hover:bg-danger-ink"
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
                      className="rounded-full border border-danger/30 bg-surface px-3.5 py-1.5 text-sm font-medium text-danger transition hover:bg-danger-wash disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Xóa{selected.size > 0 ? ` (${selected.size})` : ""}
                    </button>
                    <button
                      type="button"
                      onClick={exitSelection}
                      className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm text-ink-soft transition hover:bg-surface-2"
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

      {sources.length === 0 && (
        <div className="mt-10 flex flex-col items-center rounded-[var(--radius-card)] border border-dashed border-line px-6 py-16 text-center">
          <Waveform bars={36} className="h-12 w-56" />
          <p className="mt-6 font-display text-lg font-semibold text-ink">Chưa có tài liệu nào</p>
          <p className="mt-1 max-w-sm text-sm text-ink-soft">
            Ghi âm một cuộc họp hoặc tải file lên ở phía trên. Tài liệu sẽ xuất hiện tại đây.
          </p>
        </div>
      )}

      <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {sources.map((s, i) => (
          <div
            key={s.id}
            // The first (latest) file gets a wide slot so the grid has a focal point.
            className={`animate-enter ${i === 0 ? "sm:col-span-2" : ""}`}
            // Stagger the first rows only; later cards appear at once.
            style={{ animationDelay: `${Math.min(i, 8) * 40}ms` }}
          >
            <SourceCard
              featured={i === 0}
              source={s}
              selectionMode={selectionMode}
              selected={selected.has(s.id)}
              onToggleSelect={() => toggle(s.id)}
            />
          </div>
        ))}
      </div>
    </>
  );
}

function StatTile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="px-5 py-4">
      <dt className="text-xs font-medium text-ink-faint">{label}</dt>
      <dd className="mt-1.5 font-mono text-2xl font-medium tracking-tight tabular-nums text-ink">{value}</dd>
      <dd className="mt-0.5 text-xs text-ink-soft">{detail}</dd>
    </div>
  );
}
