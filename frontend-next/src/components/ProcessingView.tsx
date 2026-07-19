"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { Source } from "@/lib/types";
import { MOCK_SECTIONS, MOCK_TRANSCRIPT } from "@/lib/mock";
import { formatStamp } from "@/lib/format";
import { SpeakerChip, StatusBadge, Waveform } from "@/components/ui";
import { StageProgress } from "@/components/StageProgress";
import { MarkdownLite } from "@/components/MarkdownLite";
import { FinalizeDialog } from "@/components/FinalizeDialog";

// Base build: simulate the realtime stream so the team can see the intended UX.
// Real build: drive this from the `useJobEvents` WebSocket hook (doc 12 §4).
export function ProcessingView({ source }: { source: Source }) {
  const [nSeg, setNSeg] = useState(2);
  const [nSec, setNSec] = useState(2);
  const [done, setDone] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const transcriptEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (nSeg >= MOCK_TRANSCRIPT.length && nSec >= MOCK_SECTIONS.length) {
      const t = setTimeout(() => setDone(true), 1200);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setNSeg((n) => Math.min(n + 1, MOCK_TRANSCRIPT.length));
      if (nSeg % 2 === 0) setNSec((n) => Math.min(n + 1, MOCK_SECTIONS.length));
    }, 2200);
    return () => clearTimeout(t);
  }, [nSeg, nSec]);

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [nSeg]);

  const segments = MOCK_TRANSCRIPT.slice(0, nSeg);
  const sections = MOCK_SECTIONS.slice(0, nSec);

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
          <StatusBadge status={done ? "done" : "processing"} />
          {done ? (
            <button
              onClick={() => setDialogOpen(true)}
              className="rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-ink"
            >
              Chọn đầu ra
            </button>
          ) : (
            <span className="hidden font-mono text-sm text-signal-ink sm:inline">
              đang nghe…
            </span>
          )}
        </div>
      </div>

      {/* Stage progress */}
      <div className="mt-6 rounded-2xl border border-line bg-surface p-4">
        <StageProgress
          current={done ? "done" : "generating_doc"}
          progress={done ? 1 : Math.min(nSeg / MOCK_TRANSCRIPT.length, 0.95)}
        />
      </div>

      {/* Two columns — voice on the left, document on the right */}
      <div className="mt-5 grid min-h-0 flex-1 grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        {/* LEFT — live transcript */}
        <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-line bg-surface">
          <header className="flex items-center justify-between border-b border-line-soft px-5 py-3.5">
            <p className="eyebrow">Lời thoại</p>
            <Waveform live={!done} bars={16} className="h-4 w-24" />
          </header>
          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-5">
            {segments.map((seg, i) => {
              const last = i === segments.length - 1;
              return (
                <div key={seg.startMs} className="animate-rise">
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
          </div>
        </section>
      </div>

      <FinalizeDialog
        open={dialogOpen}
        sourceId={source.id}
        onClose={() => setDialogOpen(false)}
      />
    </div>
  );
}
