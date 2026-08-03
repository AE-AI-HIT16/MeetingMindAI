"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { Source } from "@/lib/types";
import { formatStamp } from "@/lib/format";
import { useJobEvents } from "@/lib/useJobEvents";
import { SpeakerChip, StatusBadge, Waveform } from "@/components/ui";
import { StageProgress } from "@/components/StageProgress";
import { MarkdownLite } from "@/components/MarkdownLite";
import { FinalizeDialog } from "@/components/FinalizeDialog";
import { mediaUrl } from "@/lib/api";

export function ProcessingView({
  source,
  jobId,
}: {
  source: Source;
  jobId: string | null;
}) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const {
    isConnected,
    stage,
    progress,
    segments,
    sections,
    done,
    liveDocumentId,
    error,
  } = useJobEvents(jobId);

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [segments.length]);

  return (
    <div className="mx-auto flex h-screen w-full max-w-6xl flex-col px-6 py-8 md:px-10">
      {/* Top bar */}
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <Link
            href="/"
            className="eyebrow inline-flex items-center gap-1.5 hover:text-brand"
          >
            ← Thư viện
          </Link>
          <h1 className="font-display mt-1.5 truncate text-2xl font-semibold text-ink">
            {source.title}
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge
            status={error ? "failed" : done ? "done" : "processing"}
          />
          {done ? (
            <button
              onClick={() => setDialogOpen(true)}
              className="rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-ink"
            >
              Chọn đầu ra
            </button>
          ) : (
            <span className="hidden font-mono text-sm text-signal-ink sm:inline">
              {isConnected ? "đang nhận dữ liệu…" : "đang kết nối…"}
            </span>
          )}
        </div>
      </div>

      {/* Stage progress */}
      <div className="mt-6 rounded-2xl border border-line bg-surface p-4">
        <StageProgress
          current={stage}
          progress={progress}
        />
      </div>

      {(done || source.status === "done") && (
        <div className="mt-4 rounded-2xl border border-line bg-surface p-4">
          <p className="eyebrow mb-3">Audio đã ghi</p>
          {source.mediaType === "video" ? (
            <video
              controls
              preload="metadata"
              className="max-h-64 w-full rounded-xl bg-black"
              src={mediaUrl(source.id)}
            />
          ) : (
            <audio
              controls
              preload="metadata"
              className="w-full"
              src={mediaUrl(source.id)}
            />
          )}
        </div>
      )}

      {error && (
        <p
          role="alert"
          className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {error}
        </p>
      )}

      {/* Two columns — voice on the left, document on the right */}
      <div className="mt-5 grid min-h-0 flex-1 grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        {/* LEFT — live transcript */}
        <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-line bg-surface">
          <header className="flex items-center justify-between border-b border-line-soft px-5 py-3.5">
            <p className="eyebrow">Lời thoại</p>
            <Waveform live={!done && !error} bars={16} className="h-4 w-24" />
          </header>
          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-5">
            {segments.map((seg, i) => {
              const last = i === segments.length - 1;
              return (
                <div
                  key={seg.id ?? `${seg.startMs}-${i}`}
                  className="animate-rise"
                >
                  <div className="mb-1 flex items-center gap-2.5">
                    <SpeakerChip speaker={seg.speaker} />
                    <span className="font-mono text-[11px] text-ink-faint">
                      {formatStamp(seg.startMs)}
                    </span>
                  </div>
                  <p
                    className={`text-[15px] leading-relaxed text-ink ${
                      last && !done ? "caret" : ""
                    }`}
                  >
                    {seg.text}
                  </p>
                </div>
              );
            })}
            {segments.length === 0 && !error && (
              <p className="text-sm text-ink-faint">
                Transcript sẽ hiện tại đây khi ASR xử lý xong đoạn đầu tiên.
              </p>
            )}
            <div ref={transcriptEnd} />
          </div>
        </section>

        {/* RIGHT — document forming */}
        <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-line bg-surface">
          <header className="flex items-center justify-between border-b border-line-soft px-5 py-3.5">
            <p className="eyebrow">Tài liệu</p>
            <span className="font-mono text-[11px] text-ink-faint">
              {sections.length} mục
            </span>
          </header>
          <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-6 py-6">
            {sections.map((sec, i) => {
              const writing = !done && i === sections.length - 1;
              return (
                <article
                  key={sec.id}
                  className="animate-rise rounded-xl p-3 -mx-3"
                  style={
                    writing
                      ? { animation: "writing 1.6s ease-in-out infinite" }
                      : undefined
                  }
                >
                  <h2 className="font-display text-lg font-semibold text-ink">
                    {sec.heading}
                  </h2>
                  <div className="mt-1.5">
                    <MarkdownLite text={sec.markdown} />
                  </div>
                </article>
              );
            })}
            {sections.length === 0 && !error && (
              <p className="text-sm text-ink-faint">
                Tài liệu trực tiếp sẽ xuất hiện sau khi nhận dạng lời nói.
              </p>
            )}
          </div>
        </section>
      </div>

      <FinalizeDialog
        open={dialogOpen}
        liveDocumentId={liveDocumentId}
        onClose={() => setDialogOpen(false)}
      />
    </div>
  );
}
