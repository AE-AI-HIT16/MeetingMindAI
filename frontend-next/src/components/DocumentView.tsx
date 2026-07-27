"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import type { DocumentData, Source } from "@/lib/types";
import { documentExportUrl, mediaUrl } from "@/lib/api";
import { formatDuration, formatStamp } from "@/lib/format";
import { useJobEvents } from "@/lib/useJobEvents";
import { MarkdownLite } from "@/components/MarkdownLite";
import { SpeakerChip } from "@/components/ui";

type Tab = "doc" | "transcript" | "media";

export function DocumentView({
  source,
  document,
}: {
  source: Source;
  document: DocumentData;
}) {
  const [tab, setTab] = useState<Tab>("doc");
  const [exportOpen, setExportOpen] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const { segments } = useJobEvents(source.jobId);

  function seekTo(startMs: number) {
    const media =
      source.mediaType === "video" ? videoRef.current : audioRef.current;
    if (!media) return;
    media.currentTime = startMs / 1000;
    void media.play().catch(() => {
      // Browser controls still allow manual playback if autoplay is blocked.
    });
  }

  return (
    <div className="mx-auto w-full max-w-4xl px-6 py-10 md:px-10">
      <Link
        href="/"
        className="eyebrow inline-flex items-center gap-1.5 hover:text-brand"
      >
        ← Thư viện
      </Link>

      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-semibold text-ink">
            {source.title}
          </h1>
          <p className="mt-2 flex items-center gap-3 text-sm text-ink-soft">
            <span className="font-mono text-xs uppercase">
              {source.mediaType}
            </span>
            <span className="text-ink-faint">·</span>
            <span className="font-mono text-xs">
              {formatDuration(source.durationMs)}
            </span>
            <span className="text-ink-faint">·</span>
            <span>
              {document.mode === "summary"
                ? "Tóm tắt bởi MeetingMind"
                : "Bản toàn văn"}
            </span>
          </p>
        </div>

        {/* Export */}
        <div className="relative">
          <button
            onClick={() => setExportOpen((v) => !v)}
            className="flex items-center gap-2 rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-ink"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3v12M8 11l4 4 4-4" />
              <path d="M4 21h16" />
            </svg>
            Xuất
          </button>
          {exportOpen && (
            <div className="absolute right-0 z-10 mt-2 w-40 overflow-hidden rounded-xl border border-line bg-surface py-1 shadow-[var(--shadow-lift)]">
              {(
                [
                  ["PDF", "pdf"],
                  ["DOCX", "docx"],
                  ["Markdown", "md"],
                ] as const
              ).map(([label, format]) => (
                <a
                  key={format}
                  href={documentExportUrl(document.id, format)}
                  onClick={() => setExportOpen(false)}
                  className="flex w-full items-center justify-between px-4 py-2 text-sm text-ink-soft hover:bg-surface-2 hover:text-ink"
                >
                  {label}
                  <span className="font-mono text-[10px] text-ink-faint">
                    .{format}
                  </span>
                </a>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="mt-7 flex gap-1 border-b border-line">
        {(
          [
            ["doc", "Tài liệu"],
            ["transcript", "Lời thoại"],
            ["media", "Media"],
          ] as [Tab, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              tab === key
                ? "border-brand text-brand-ink"
                : "border-transparent text-ink-faint hover:text-ink"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Keep one media element mounted so playback survives tab changes. */}
      <section
        aria-label="Trình phát media"
        className={
          tab === "media"
            ? "py-8"
            : "sticky top-3 z-20 mt-4 rounded-2xl border border-line bg-ink/95 p-3 shadow-[var(--shadow-lift)] backdrop-blur"
        }
      >
        {tab !== "media" && (
          <div className="mb-2 flex items-center justify-between gap-3 px-1">
            <span className="flex items-center gap-2 text-xs font-medium text-white/80">
              <span className="h-2 w-2 rounded-full bg-signal" />
              Media nguồn
            </span>
            <button
              onClick={() => setTab("media")}
              className="text-xs text-white/65 transition hover:text-white"
            >
              Mở trình phát
            </button>
          </div>
        )}

        <div
          className={`overflow-hidden bg-ink ${
            tab === "media"
              ? "rounded-2xl border border-line p-4"
              : "rounded-xl"
          }`}
        >
          {source.mediaType === "video" ? (
            <video
              ref={videoRef}
              controls
              preload="metadata"
              className={
                tab === "media"
                  ? "aspect-video w-full"
                  : "max-h-48 w-full bg-black"
              }
              src={mediaUrl(source.id)}
            />
          ) : (
            <audio
              ref={audioRef}
              controls
              preload="metadata"
              className="w-full"
              src={mediaUrl(source.id)}
            />
          )}
        </div>
      </section>

      {/* Document and transcript remain readable behind the sticky player. */}
      {tab !== "media" && (
        <div className="py-8">
          {tab === "doc" && (
            <article>
              <MarkdownLite text={document.markdown} />
            </article>
          )}

          {tab === "transcript" && (
            <div className="space-y-5">
              {segments.map((seg) => (
                <div
                  key={seg.id ?? seg.startMs}
                  className="flex gap-4"
                >
                  <button
                    type="button"
                    onClick={() => seekTo(seg.startMs)}
                    title={`Phát từ ${formatStamp(seg.startMs)}`}
                    className="w-12 shrink-0 pt-0.5 text-left font-mono text-[11px] text-ink-faint transition hover:text-brand"
                  >
                    {formatStamp(seg.startMs)}
                  </button>
                  <div>
                    <SpeakerChip speaker={seg.speaker} />
                    <p className="mt-1 text-[15px] leading-relaxed text-ink">
                      {seg.text}
                    </p>
                  </div>
                </div>
              ))}
              {segments.length === 0 && (
                <p className="text-sm text-ink-faint">
                  Đang tải lại transcript từ Job…
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
