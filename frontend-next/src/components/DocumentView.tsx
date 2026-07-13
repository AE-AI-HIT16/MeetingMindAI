"use client";

import { useState } from "react";
import Link from "next/link";
import type { Source } from "@/lib/types";
import { MOCK_SECTIONS, MOCK_TRANSCRIPT } from "@/lib/mock";
import { formatDuration, formatStamp } from "@/lib/format";
import { MarkdownLite } from "@/components/MarkdownLite";
import { SpeakerChip, Waveform } from "@/components/ui";

type Tab = "doc" | "transcript" | "media";

export function DocumentView({ source }: { source: Source }) {
  const [tab, setTab] = useState<Tab>("doc");
  const [exportOpen, setExportOpen] = useState(false);

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
            <span>Tóm tắt bởi MeetingMind</span>
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
              {["PDF", "DOCX", "Markdown"].map((f) => (
                <button
                  key={f}
                  onClick={() => setExportOpen(false)}
                  className="flex w-full items-center justify-between px-4 py-2 text-sm text-ink-soft hover:bg-surface-2 hover:text-ink"
                >
                  {f}
                  <span className="font-mono text-[10px] text-ink-faint">
                    .{f.toLowerCase().slice(0, 3)}
                  </span>
                </button>
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

      {/* Panels */}
      <div className="py-8">
        {tab === "doc" && (
          <article className="space-y-7">
            {MOCK_SECTIONS.map((sec) => (
              <section key={sec.id}>
                <h2 className="font-display text-xl font-semibold text-ink">
                  {sec.heading}
                </h2>
                <div className="mt-2">
                  <MarkdownLite text={sec.markdown} />
                </div>
              </section>
            ))}
          </article>
        )}

        {tab === "transcript" && (
          <div className="space-y-5">
            {MOCK_TRANSCRIPT.map((seg) => (
              <div key={seg.startMs} className="flex gap-4">
                <span className="w-12 shrink-0 pt-0.5 font-mono text-[11px] text-ink-faint">
                  {formatStamp(seg.startMs)}
                </span>
                <div>
                  <SpeakerChip speaker={seg.speaker} />
                  <p className="mt-1 text-[15px] leading-relaxed text-ink">
                    {seg.text}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

        {tab === "media" && (
          <div className="overflow-hidden rounded-2xl border border-line bg-ink">
            <div className="flex aspect-video items-center justify-center bg-gradient-to-br from-ink to-[#26263a]">
              <Waveform bars={48} className="h-20 w-3/4 opacity-40" />
            </div>
            <div className="flex items-center gap-3 bg-surface px-4 py-3">
              <button className="flex h-9 w-9 items-center justify-center rounded-full bg-brand text-white">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
              <div className="h-1 flex-1 rounded-full bg-line">
                <div className="h-full w-1/4 rounded-full bg-brand" />
              </div>
              <span className="font-mono text-xs text-ink-faint">
                7:04 / {formatDuration(source.durationMs)}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
