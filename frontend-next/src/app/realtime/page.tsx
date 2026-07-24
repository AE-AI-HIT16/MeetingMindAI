"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRealtimeStream } from "@/lib/useRealtimeStream";
import type { AudioSource, SentenceInfo } from "@/lib/useRealtimeStream";
import { formatDuration } from "@/lib/format";
import { StatusBadge, Waveform } from "@/components/ui";

// ----------------------------------------------------------------
// Page
// ----------------------------------------------------------------

export default function RealtimePage() {
  const {
    isRecording,
    isConnected,
    transcripts,
    elapsedMs,
    error,
    start,
    stop,
  } = useRealtimeStream();

  const scrollRef = useRef<HTMLDivElement>(null);
  const [audioSource, setAudioSource] = useState<AudioSource>("microphone");

  // Auto-scroll transcript panel
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcripts.length]);

  return (
    <div className="mx-auto flex h-screen w-full max-w-5xl flex-col px-6 py-8 md:px-10">
      {/* ---- Top bar ---- */}
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <Link
            href="/"
            className="eyebrow inline-flex items-center gap-1.5 hover:text-brand"
          >
            ← Thư viện
          </Link>
          <h1 className="font-display mt-1.5 text-2xl font-semibold text-ink">
            Ghi âm trực tiếp
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={isRecording ? "processing" : "done"} />
          {isRecording && (
            <span className="hidden font-mono text-sm text-signal-ink sm:inline">
              {formatDuration(elapsedMs)}
            </span>
          )}
        </div>
      </div>

      {/* ---- Control panel ---- */}
      <div className="mt-6 rounded-2xl border border-line bg-surface p-6">
        <div className="flex flex-col items-center gap-5 sm:flex-row sm:justify-between">
          {/* Left: info */}
          <div className="text-center sm:text-left">
            <p className="font-display text-lg font-semibold text-ink">
              {isRecording
                ? "Đang ghi âm và nhận dạng…"
                : transcripts.length > 0
                  ? "Ghi âm đã kết thúc"
                  : "Nhấn nút để bắt đầu ghi âm"}
            </p>
            <p className="mt-1 text-sm text-ink-soft">
              {isRecording
                ? "Audio được stream realtime tới server, transcript hiện dần bên dưới."
                : "Microphone → WebSocket → ASR pipeline → Transcript"}
            </p>
            {error && (
              <p className="mt-2 text-sm font-medium text-danger">{error}</p>
            )}
          </div>

          {/* Right: record button */}
          <div className="flex items-center gap-4">
            {/* Waveform indicator */}
            <Waveform live={isRecording} bars={20} className="h-10 w-28" />

            {/* Big record / stop button */}
            <button
              id="record-btn"
              onClick={isRecording ? stop : () => start(audioSource)}
              className="group relative flex h-16 w-16 items-center justify-center rounded-full transition-shadow"
              style={{
                background: isRecording
                  ? "var(--color-signal)"
                  : "var(--color-brand)",
                boxShadow: isRecording
                  ? "0 0 0 4px var(--color-signal-wash)"
                  : "0 2px 8px rgba(44,62,224,0.25)",
              }}
              aria-label={isRecording ? "Dừng ghi âm" : "Bắt đầu ghi âm"}
            >
              {/* Expanding ring when recording */}
              {isRecording && <span className="recording-ring absolute inset-0 rounded-full" />}

              {isRecording ? (
                /* Stop icon */
                <svg
                  width="22"
                  height="22"
                  viewBox="0 0 24 24"
                  fill="white"
                  className="transition group-hover:scale-110"
                >
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              ) : (
                /* Mic icon */
                <svg
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="white"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="transition group-hover:scale-110"
                >
                  <rect x="9" y="2" width="6" height="12" rx="3" />
                  <path d="M5 10a7 7 0 0 0 14 0" />
                  <line x1="12" y1="19" x2="12" y2="22" />
                  <line x1="8" y1="22" x2="16" y2="22" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {!isRecording && (
          <div className="mt-5 flex flex-wrap gap-2" role="radiogroup" aria-label="Nguồn âm thanh">
            <SourceOption label="Microphone" selected={audioSource === "microphone"} onClick={() => setAudioSource("microphone")} />
            <SourceOption label="Âm thanh từ tab" selected={audioSource === "tab"} onClick={() => setAudioSource("tab")} />
          </div>
        )}

        {/* Connection status bar */}
        {isRecording && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2">
            <span
              className={`h-2 w-2 rounded-full ${
                isConnected ? "bg-ok" : "bg-danger recording-dot"
              }`}
            />
            <span className="font-mono text-xs text-ink-faint">
              {isConnected
                ? "WebSocket kết nối · PCM16 mono 16kHz"
                : "Đang kết nối…"}
            </span>
            <span className="ml-auto font-mono text-xs text-ink-faint">
              {transcripts.length} segment(s)
            </span>
          </div>
        )}
      </div>

      {/* ---- Transcript panel ---- */}
      <section className="mt-5 flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-line bg-surface">
        <header className="flex items-center justify-between border-b border-line-soft px-5 py-3.5">
          <p className="eyebrow">Transcript Realtime</p>
          <Waveform live={isRecording} bars={16} className="h-4 w-24" />
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          {transcripts.length === 0 ? (
            <EmptyState isRecording={isRecording} />
          ) : (
            <div className="space-y-4">
              {transcripts.map((t, i) => (
                <TranscriptBlock
                  key={i}
                  text={t.text}
                  sentenceInfo={t.sentenceInfo}
                  index={i}
                  isLast={i === transcripts.length - 1}
                  isRecording={isRecording}
                  receivedAt={t.receivedAt}
                />
              ))}
              <div ref={scrollRef} />
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

// ----------------------------------------------------------------
// Sub-components
// ----------------------------------------------------------------

function EmptyState({ isRecording }: { isRecording: boolean }) {
  return (
    <div className="flex h-full flex-col items-center justify-center py-16 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-brand-wash">
        {isRecording ? (
          <span className="h-3 w-3 rounded-full bg-signal recording-dot" />
        ) : (
          <svg
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--color-brand)"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="9" y="2" width="6" height="12" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0" />
            <line x1="12" y1="19" x2="12" y2="22" />
            <line x1="8" y1="22" x2="16" y2="22" />
          </svg>
        )}
      </div>
      <p className="font-display mt-4 text-lg font-medium text-ink">
        {isRecording
          ? "Đang chờ backend trả transcript…"
          : "Chưa có transcript"}
      </p>
      <p className="mt-1.5 max-w-xs text-sm text-ink-soft">
        {isRecording
          ? "Audio đang được stream tới server. Transcript sẽ xuất hiện khi VAD phát hiện đủ speech (~4 giây)."
          : "Nhấn nút ghi âm phía trên để bắt đầu stream audio và nhận transcript thời gian thực."}
      </p>
    </div>
  );
}

function SourceOption({ label, selected, onClick }: { label: string; selected: boolean; onClick: () => void }) {
  return (
    <button type="button" role="radio" aria-checked={selected} onClick={onClick}
      className={`rounded-lg border px-3 py-2 text-sm font-medium transition ${selected ? "border-brand bg-brand-wash text-brand-ink" : "border-line text-ink-soft hover:border-brand"}`}>
      {label}
    </button>
  );
}

function TranscriptBlock({
  text,
  sentenceInfo,
  index,
  isLast,
  isRecording,
  receivedAt,
}: {
  text: string;
  sentenceInfo: SentenceInfo[];
  index: number;
  isLast: boolean;
  isRecording: boolean;
  receivedAt: number;
}) {
  const timeStr = new Date(receivedAt).toLocaleTimeString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="animate-transcript">
      <div className="flex items-start gap-3">
        {/* Index badge */}
        <span
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold text-white"
          style={{
            background:
              "linear-gradient(135deg, var(--color-brand), var(--color-spk-4))",
          }}
        >
          {index + 1}
        </span>

        <div className="min-w-0 flex-1">
          {/* Timestamp */}
          <span className="font-mono text-[11px] text-ink-faint">
            {timeStr}
          </span>

          {sentenceInfo.length > 0 ? (
            <div className="mt-1 space-y-2">
              {sentenceInfo.map((sentence, sentenceIndex) => (
                <p key={`${sentence.start}-${sentenceIndex}`} className="text-[15px] leading-relaxed text-ink">
                  <span className="mr-2 rounded bg-brand-wash px-1.5 py-0.5 text-xs font-semibold text-brand-ink">
                    {sentence.speaker === null || sentence.speaker === undefined ? "Speaker" : `Speaker ${sentence.speaker}`}
                  </span>
                  {sentence.text}
                </p>
              ))}
            </div>
          ) : (
            <p className={`mt-0.5 text-[15px] leading-relaxed text-ink ${isLast && isRecording ? "caret" : ""}`}>{text}</p>
          )}
        </div>
      </div>

      {/* Separator */}
      {!isLast && (
        <div className="ml-10 mt-3 border-b border-line-soft" />
      )}
    </div>
  );
}
