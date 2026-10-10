"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRealtimeStream } from "@/lib/useRealtimeStream";
import { useJobEvents } from "@/lib/useJobEvents";
import { EditableSegmentText } from "@/components/EditableSegmentText";
import type { AudioSource, SentenceInfo } from "@/lib/useRealtimeStream";
import {
  isTrackCaptionSupported,
  isWebSpeechSupported,
  useWebSpeechCaption,
} from "@/lib/useWebSpeechCaption";
import { groupCaptionParagraphs, unconfirmedCaptions } from "@/lib/liveCaptionState";
import type { CaptionParagraph } from "@/lib/liveCaptionState";
import { formatDuration, formatStamp, getSpeakerStyle } from "@/lib/format";
import { PageTitle, SpeakerChip, StatusBadge, Waveform } from "@/components/ui";
import { ErrorNotice } from "@/components/ErrorNotice";
import {
  ArrowLeft,
  ArrowRight,
  Browser,
  FileText,
  Microphone,
  Stop,
} from "@phosphor-icons/react";
import { EMPTY_HINTS, RecognitionHints } from "@/components/RecognitionHints";

// ----------------------------------------------------------------
// Page
// ----------------------------------------------------------------

const noopSubscribe = () => () => {};

export default function RealtimePage() {
  const {
    isRecording,
    isFinalizing,
    isConnected,
    transcripts,
    elapsedMs,
    error,
    sourceId,
    jobId,
    confirmedEndMs,
    start,
    stop,
  } = useRealtimeStream();

  const router = useRouter();
  const caption = useWebSpeechCaption("vi-VN");

  const scrollRef = useRef<HTMLDivElement>(null);
  const [audioSource, setAudioSource] = useState<AudioSource>("microphone");
  const [hints, setHints] = useState(EMPTY_HINTS);
  // Client-only detection (false during SSR) to avoid hydration mismatch.
  const captionSupported = useSyncExternalStore(
    noopSubscribe,
    isWebSpeechSupported,
    () => false,
  );
  const trackCaptionSupported = useSyncExternalStore(
    noopSubscribe,
    isTrackCaptionSupported,
    () => false,
  );
  const [captionActive, setCaptionActive] = useState(false);

  // The system picks the mode; the user only presses record:
  //  - Chrome/Edge + mic, or Chrome 135+ + tab audio: browser Web Speech shows
  //    faded words instantly while RunPod (cold-started on connect) confirms
  //    each 15 s window in solid text.
  //  - Otherwise (Brave/Firefox/older Chrome tab audio): server only, shorter
  //    10 s windows so the first solid text arrives sooner.
  const handleStart = async () => {
    const useCaption =
      audioSource === "microphone" ? captionSupported : trackCaptionSupported;
    const started = await start(audioSource, {
      windowSeconds: useCaption ? 15 : 10,
      context: hints.context,
    });
    setCaptionActive(useCaption && started !== null);
    if (useCaption && started !== null) {
      // Same track the server receives (required for tab audio).
      caption.start(
        started.startedAt,
        trackCaptionSupported ? started.audioTrack : undefined,
      );
    }
  };

  const handleStop = () => {
    caption.stop();
    stop();
  };

  // Recording ended for any reason (stop, WS error) → stop captions too.
  const stopCaption = caption.stop;
  useEffect(() => {
    if (!isRecording) stopCaption();
  }, [isRecording, stopCaption]);

  const interimCaption = captionActive ? caption.interim : null;
  const captionParagraphs = captionActive
    ? groupCaptionParagraphs(
        unconfirmedCaptions(caption.lines, confirmedEndMs),
        interimCaption,
      )
    : [];
  const hasContent = transcripts.length > 0 || captionParagraphs.length > 0;

  // After stop, the offline job (VAD + speaker diarization + ASR) starts right
  // away on the still-warm RunPod worker. Follow it here and, when it is done,
  // swap the realtime text for the speaker-labelled transcript.
  const followJobId = !isRecording && !isFinalizing && hasContent ? jobId : null;
  const job = useJobEvents(followJobId);
  const diarized = followJobId && job.done && job.segments.length > 0 ? job.segments : null;
  const diarizing = followJobId !== null && !job.done && !job.error;

  // Auto-scroll transcript panel
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcripts.length, captionParagraphs.length]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ block: "end" });
  }, [interimCaption?.text]);

  const statusTitle = isFinalizing
    ? "Đang hoàn tất phần audio cuối…"
    : isRecording
      ? "Đang ghi âm"
      : hasContent
        ? "Ghi âm đã kết thúc"
        : "Sẵn sàng ghi âm";
  const statusDetail = isFinalizing
    ? "Máy chủ đang chốt phần lời thoại cuối. Vui lòng giữ trang này mở."
    : isRecording
      ? "Lời thoại hiện dần ở bên cạnh."
      : "Lời thoại hiện ngay khi bạn nói, sau đó được tách theo người nói.";

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col px-4 py-6 sm:px-6 md:h-dvh md:px-10 md:py-8">
      {/* ---- Top bar ---- */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <Link
            href="/"
            className="eyebrow inline-flex items-center gap-1.5 transition hover:text-brand-ink"
          >
            <ArrowLeft size={12} />
            Thư viện
          </Link>
          <div className="mt-2">
            <PageTitle title="Ghi âm trực tiếp" lede="Bạn nói tới đâu, chữ hiện tới đó." />
          </div>
        </div>
        {(isRecording || isFinalizing || hasContent) && (
          <StatusBadge status={isRecording || isFinalizing ? "processing" : "done"} />
        )}
      </div>

      <div className="mt-6 grid min-h-0 flex-1 gap-5 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        {/* ---- Console: the record control and its settings ---- */}
        <div className="flex min-h-0 flex-col gap-4 lg:overflow-y-auto">
          <div className={`bezel transition-colors ${isRecording ? "bg-signal/10" : ""}`}>
            <div className="bezel-core relative overflow-hidden p-6">
              <span
                aria-hidden="true"
                className={`pointer-events-none absolute left-1/2 top-24 h-64 w-64 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand/15 blur-3xl transition-opacity duration-700 ${
                  isRecording ? "opacity-100" : "opacity-30"
                }`}
              />

              <div className="relative flex flex-col items-center text-center">
                {/* Big record / stop button */}
                <button
                  id="record-btn"
                  onClick={isRecording ? handleStop : handleStart}
                  disabled={isFinalizing}
                  className={`group relative mt-2 flex h-24 w-24 shrink-0 items-center justify-center rounded-full text-on-brand shadow-[inset_0_1px_0_rgb(255_255_255/0.3),0_18px_40px_-16px_rgb(242_114_79/0.7)] transition disabled:cursor-wait disabled:opacity-60 ${
                    isRecording ? "bg-signal" : "bg-brand hover:bg-brand-ink"
                  }`}
                  aria-label={
                    isFinalizing
                      ? "Đang hoàn tất audio"
                      : isRecording
                        ? "Dừng ghi âm"
                        : "Bắt đầu ghi âm"
                  }
                >
                  {/* Expanding ring when recording */}
                  {isRecording && <span className="recording-ring absolute inset-0 rounded-full" />}
                  {isRecording ? (
                    <Stop size={30} weight="fill" className="transition group-hover:scale-110" />
                  ) : (
                    <Microphone size={34} weight="fill" className="transition group-hover:scale-110" />
                  )}
                </button>

                <p
                  className={`mt-6 font-mono text-4xl font-medium tabular-nums tracking-tight ${
                    isRecording ? "text-ink" : "text-ink-faint"
                  }`}
                  aria-live="off"
                >
                  {formatDuration(elapsedMs)}
                </p>
                <Waveform live={isRecording} bars={36} className="mt-5 h-10 w-full max-w-[260px]" />

                <p className="mt-5 font-display text-lg font-semibold text-ink">{statusTitle}</p>
                <p className="mt-1 max-w-[280px] text-sm text-ink-soft">{statusDetail}</p>
              </div>

              {/* Connection status */}
              {(isRecording || isFinalizing) && (
                <div className="relative mt-5 flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-2">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      isConnected ? "bg-ok" : "bg-danger recording-dot"
                    }`}
                  />
                  <span className="text-xs text-ink-soft">
                    {isFinalizing
                      ? "Đang chờ máy chủ xử lý phần cuối…"
                      : isConnected
                        ? "Đã kết nối máy chủ"
                        : "Đang kết nối…"}
                  </span>
                  <span className="ml-auto font-mono text-xs tabular-nums text-ink-faint">
                    {transcripts.length} đoạn
                  </span>
                </div>
              )}

              {error ? (
                <ErrorNotice error={error} title="Ghi âm gặp sự cố" className="relative mt-4" />
              ) : null}
            </div>
          </div>

          {!isRecording && !isFinalizing && (
            <div className="rounded-[var(--radius-bezel)] bg-surface p-6 shadow-[var(--shadow-card)] ring-1 ring-line/80">
              <p id="audio-source-label" className="text-sm font-medium text-ink">
                Nguồn âm thanh
              </p>
              <div
                className="mt-2 grid grid-cols-2 gap-1 rounded-xl bg-surface-2 p-1 ring-1 ring-line-soft"
                role="radiogroup"
                aria-labelledby="audio-source-label"
              >
                <SourceOption
                  label="Micro"
                  icon={<Microphone size={16} />}
                  selected={audioSource === "microphone"}
                  onClick={() => setAudioSource("microphone")}
                />
                <SourceOption
                  label="Âm thanh từ tab"
                  icon={<Browser size={16} />}
                  selected={audioSource === "tab"}
                  onClick={() => setAudioSource("tab")}
                />
              </div>
              <RecognitionHints value={hints} onChange={setHints} className="mt-5" />
            </div>
          )}

          {/* ---- Navigation to document page after recording ---- */}
          {!isRecording && !isFinalizing && hasContent && sourceId && (
            <div className="rounded-[var(--radius-card)] border border-brand/25 bg-brand-wash p-5">
              <p className="font-display text-base font-semibold text-ink">
                Ghi âm đã được lưu và đang được xử lý.
              </p>
              <p className="mt-1 text-sm text-ink-soft">
                Mở trang tài liệu để xem và xuất kết quả.
              </p>
              <button
                id="view-document-btn"
                onClick={() => router.push(`/sources/${sourceId}`)}
                className="btn btn-primary mt-4 w-full"
              >
                <FileText size={18} />
                Xem tài liệu
                <span className="btn-nest">
                  <ArrowRight size={14} weight="bold" />
                </span>
              </button>
            </div>
          )}
        </div>

      {/* ---- Transcript panel ---- */}
      <section className="flex min-h-[60dvh] flex-col overflow-hidden rounded-[var(--radius-bezel)] bg-surface shadow-[var(--shadow-card)] ring-1 ring-line/80 lg:min-h-0">
        <header className="flex items-center justify-between border-b border-line-soft px-5 py-3.5">
          <p className="eyebrow">Lời thoại</p>
          <Waveform live={isRecording} bars={16} className="h-4 w-24" />
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
          {!hasContent ? (
            <EmptyState isRecording={isRecording} captionActive={captionActive} />
          ) : (
            <div className="space-y-4">
              {diarized?.map((segment, i) => (
                <TranscriptBlock
                  key={segment.id ?? `${segment.startMs}-${i}`}
                  text={segment.text}
                  sentenceInfo={[
                    {
                      text: segment.text,
                      start: segment.startMs / 1000,
                      end: segment.endMs / 1000,
                      speaker: segment.speaker,
                    },
                  ]}
                  index={i}
                  isLast={i === diarized.length - 1}
                  isRecording={false}
                  type="transcript_delta"
                  speaker={segment.speaker}
                  startMs={segment.startMs}
                  endMs={segment.endMs}
                  editJobId={followJobId}
                  editSegmentId={segment.id}
                />
              ))}
              {!diarized && transcripts.map((t, i) => (
                <TranscriptBlock
                  key={`${t.startMs}-${i}`}
                  text={t.text}
                  sentenceInfo={t.sentenceInfo}
                  index={i}
                  isLast={i === transcripts.length - 1}
                  isRecording={isRecording}
                  receivedAt={t.receivedAt}
                  type={t.type}
                  speaker={t.speaker}
                  startMs={t.startMs}
                  endMs={t.endMs}
                />
              ))}
              {captionParagraphs.map((paragraph, i) => (
                !diarized && <CaptionParagraphBlock
                  key={paragraph.id}
                  paragraph={paragraph}
                  showSeparator={transcripts.length > 0 || i > 0}
                />
              ))}
              {(isRecording || isFinalizing) && !interimCaption && (
                <ListeningRow finalizing={isFinalizing} />
              )}
              {diarizing && (
                <ListeningRow
                  label={`Đang phân biệt người nói… ${Math.round(job.progress * 100)}%`}
                />
              )}
              <div ref={scrollRef} />
            </div>
          )}
        </div>
      </section>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// Sub-components
// ----------------------------------------------------------------

function EmptyState({
  isRecording,
  captionActive,
}: {
  isRecording: boolean;
  captionActive: boolean;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center py-16 text-center">
      <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-surface-2 ring-1 ring-line">
        {isRecording ? (
          <span className="h-3 w-3 rounded-full bg-signal recording-dot" />
        ) : (
<Microphone size={28} className="text-brand" />
        )}
      </div>
      <p className="font-display mt-4 text-lg font-medium text-ink">
        {isRecording
          ? "Đang chờ lời thoại đầu tiên…"
          : "Chưa có lời thoại"}
      </p>
      <p className="mt-1.5 max-w-xs text-sm text-ink-soft">
        {isRecording
          ? captionActive
            ? "Hãy bắt đầu nói. Phụ đề nhanh sẽ hiện ngay từng chữ."
            : "Audio đang được gửi tới máy chủ. Lời thoại hiện theo từng đoạn khoảng 30 giây."
          : "Nhấn nút ghi âm để bắt đầu. Lời thoại sẽ hiện tại đây."}
      </p>
    </div>
  );
}

function SourceOption({
  label,
  icon,
  selected,
  onClick,
}: {
  label: string;
  icon: React.ReactNode;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onClick}
      className={`flex items-center justify-center gap-2 whitespace-nowrap rounded-lg px-2 py-2 text-sm font-medium transition ${
        selected
          ? "bg-surface-3 text-ink shadow-[inset_0_1px_0_rgb(255_255_255/0.06)] ring-1 ring-line"
          : "text-ink-faint hover:text-ink"
      }`}
    >
      <span className={selected ? "text-brand" : undefined}>{icon}</span>
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
  type,
  speaker,
  startMs,
  endMs,
  editJobId = null,
  editSegmentId = null,
}: {
  text: string;
  sentenceInfo: SentenceInfo[];
  index: number;
  isLast: boolean;
  isRecording: boolean;
  receivedAt?: number;
  type: "transcript_delta" | "transcript_partial";
  speaker: number | null;
  startMs: number;
  endMs: number;
  /** Persisted segment (final, diarized transcript): text can be edited. */
  editJobId?: string | null;
  editSegmentId?: number | null;
}) {
  const timeStr =
    receivedAt === undefined
      ? null
      : new Date(receivedAt).toLocaleTimeString("vi-VN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });

  const isPartial = type === "transcript_partial";

  return (
    <div className="animate-transcript">
      <div className="flex items-start gap-3">
        {/* Index / Speaker Badge */}
        {isPartial ? (
          <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-surface-2 text-xs font-bold text-ink-faint">
            …
          </span>
        ) : (
          <span
            className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-bold text-on-brand"
            style={{ backgroundColor: getSpeakerStyle(speaker).color }}
          >
            {speaker !== null && speaker !== undefined ? speaker + 1 : index + 1}
          </span>
        )}

        <div className="min-w-0 flex-1">
          {/* Header row: timestamp & status badge */}
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] text-ink-faint">
              {formatStamp(startMs)} - {formatStamp(endMs)}
              {timeStr && ` (${timeStr})`}
            </span>
            {isPartial && (
              <span className="rounded-md border border-line bg-surface-2 px-1.5 py-0.5 text-[11px] font-medium text-ink-faint">
                Tạm thời
              </span>
            )}
          </div>

          {/* Transcript Content */}
          <div className="mt-1">
            {isPartial ? (
              /* Gray text for transcript_partial */
              <p className="text-[15px] leading-relaxed text-ink-faint italic">
                {text}
              </p>
            ) : (
              /* Committed transcript_delta with speaker color coding */
              <div className="space-y-1.5">
                {sentenceInfo.length > 0 ? (
                  sentenceInfo.map((sentence, sIdx) => {
                    const spkNum =
                      typeof sentence.speaker === "number"
                        ? sentence.speaker
                        : speaker;
                    const badge = <SpeakerChip speaker={spkNum} />;
                    if (editSegmentId !== null) {
                      return (
                        <div key={`${sentence.start}-${sIdx}`}>
                          <span className="mr-2">{badge}</span>
                          <EditableSegmentText
                            jobId={editJobId}
                            segmentId={editSegmentId}
                            text={sentence.text}
                            className="mt-1 text-[15px]"
                          />
                        </div>
                      );
                    }
                    return (
                      <p
                        key={`${sentence.start}-${sIdx}`}
                        className={`text-[15px] leading-relaxed text-ink ${
                          isLast && isRecording ? "caret" : ""
                        }`}
                      >
                        <span className="mr-2">{badge}</span>
                        {sentence.text}
                      </p>
                    );
                  })
                ) : (
                  <p
                    className={`text-[15px] leading-relaxed text-ink ${
                      isLast && isRecording ? "caret" : ""
                    }`}
                  >
                    <span className="mr-2">
                      <SpeakerChip speaker={speaker} />
                    </span>
                    {text}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Separator */}
      {!isLast && <div className="ml-10 mt-3 border-b border-line-soft" />}
    </div>
  );
}

function CaptionParagraphBlock({
  paragraph,
  showSeparator,
}: {
  paragraph: CaptionParagraph;
  showSeparator: boolean;
}) {
  // Same layout as a confirmed TranscriptBlock, but faded: the server replaces
  // it with solid, speaker-labelled text once this range is confirmed.
  return (
    <div className="animate-transcript">
      {showSeparator && <div className="mb-4 ml-10 border-b border-line-soft" />}
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-surface-2 text-xs font-bold text-ink-faint">
          …
        </span>
        <div className="min-w-0 flex-1">
          <span className="font-mono text-[11px] text-ink-faint">
            {formatStamp(paragraph.startMs)} - {formatStamp(paragraph.endMs)}
          </span>
          <p className="mt-1 text-[15px] leading-relaxed text-ink-faint">
            {paragraph.text}
            {paragraph.interim && (
              <span className="caret">
                {paragraph.text ? " " : ""}
                {paragraph.interim}
              </span>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

function ListeningRow({
  finalizing = false,
  label,
}: {
  finalizing?: boolean;
  label?: string;
}) {
  return (
    <div className="flex items-center gap-2 text-sm text-ink-faint" aria-live="polite">
      <span className="flex gap-1">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current [animation-delay:150ms]" />
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current [animation-delay:300ms]" />
      </span>
      {label ?? (finalizing ? "Đang hoàn thiện transcript…" : "Đang nghe…")}
    </div>
  );
}
