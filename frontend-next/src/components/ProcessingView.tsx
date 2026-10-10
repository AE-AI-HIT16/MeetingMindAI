"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { Source } from "@/lib/types";
import { formatDuration, formatStamp, getSpeakerStyle } from "@/lib/format";
import { useJobEvents } from "@/lib/useJobEvents";
import { useRevealCount } from "@/lib/useRevealCount";
import { CrosstalkBadge, SpeakerChip, StatusBadge, Waveform } from "@/components/ui";
import { groupSpeakerBlocks } from "@/lib/speakerBlocks";
import { StageProgress } from "@/components/StageProgress";
import { MarkdownLite } from "@/components/MarkdownLite";
import { FinalizeDialog } from "@/components/FinalizeDialog";
import { mediaUrl } from "@/lib/api";
import { ErrorNotice } from "@/components/ErrorNotice";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  MagicWand,
  Quotes,
  WarningCircle,
  Waveform as WaveformIcon,
  type Icon,
} from "@phosphor-icons/react";
import type { ProcessingStage } from "@/lib/types";

export function ProcessingView({
  source,
  jobId,
}: {
  source: Source;
  jobId: string | null;
}) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const transcriptScroll = useRef<HTMLDivElement>(null);
  const {
    isConnected,
    stage,
    progress,
    segments,
    snapshotCount,
    sections,
    done,
    liveDocumentId,
    error,
  } = useJobEvents(jobId);

  // Reveal batched segments one by one so the transcript keeps flowing.
  const visibleCount = useRevealCount(segments.length, snapshotCount);
  const visibleSegments = segments.slice(0, visibleCount);

  useEffect(() => {
    // Scroll only the transcript panel: on mobile the page itself scrolls,
    // and scrollIntoView would yank the whole window down.
    const panel = transcriptScroll.current;
    panel?.scrollTo({ top: panel.scrollHeight, behavior: "smooth" });
  }, [visibleCount]);

  const headline = error
    ? { title: "Xử lý bị gián đoạn.", muted: "Xem chi tiết bên dưới." }
    : done
      ? { title: "Xong rồi.", muted: "Chọn dạng tài liệu bạn muốn." }
      : STAGE_COPY[stage === "done" ? "generating_doc" : stage];
  const stageIndex = STAGE_ORDER.indexOf(stage === "done" ? "generating_doc" : stage);
  const overallPercent = Math.round(
    ((Math.max(stageIndex, 0) + progress) / STAGE_ORDER.length) * 100,
  );

  const hasMedia = done || source.status === "done";

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 md:px-10 md:py-10">
      {/* Header */}
      <div className="animate-enter flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <Link
            href="/"
            className="eyebrow inline-flex items-center gap-1.5 transition hover:text-brand-ink"
          >
            <ArrowLeft size={12} />
            Thư viện
          </Link>
          <h1 className="mt-2 truncate font-display text-3xl font-semibold tracking-[-0.03em] text-ink md:text-4xl">
            {source.title}
          </h1>
          <p className="mt-2 flex items-center gap-3 font-mono text-xs text-ink-faint">
            <span className="uppercase">{source.mediaType}</span>
            <span>{formatDuration(source.durationMs)}</span>
            <StatusBadge status={error ? "failed" : done ? "done" : "processing"} />
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          {done ? (
            <button
              onClick={() => setDialogOpen(true)}
              className="btn btn-primary animate-attention"
            >
              Chọn đầu ra
              <span className="btn-nest">
                <ArrowRight size={14} weight="bold" />
              </span>
            </button>
          ) : (
            <span className="font-mono text-xs text-signal-ink">
              {isConnected ? "đang nhận dữ liệu…" : "đang kết nối…"}
            </span>
          )}
        </div>
      </div>

      {/* Status: live orb, what the machine is doing, stages; the player
          joins this card once the audio is ready. */}
      <div className="bezel animate-enter mt-7 [animation-delay:80ms]">
        <div className="bezel-core relative overflow-hidden p-5 sm:p-6">
          <span
            aria-hidden="true"
            className={`pointer-events-none absolute -left-16 -top-24 h-64 w-64 rounded-full blur-3xl transition-colors duration-700 ${
              error ? "bg-danger/10" : done ? "bg-ok/15" : "bg-brand/15"
            }`}
          />
          <div className="relative grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:items-center">
            <div className="flex items-center gap-5">
              <StatusOrb stage={stage} done={done} failed={!!error} />
              <div className="min-w-0 flex-1" aria-live="polite">
                <p className="font-display text-2xl font-semibold tracking-[-0.03em] text-ink">
                  {headline.title}
                </p>
                <p className="mt-0.5 text-sm text-ink-soft">
                  {headline.muted}
                  {!done && !error && (
                    <span className="ml-2 font-mono text-xs tabular-nums text-ink-faint">
                      {overallPercent}%
                    </span>
                  )}
                </p>
              </div>
            </div>
            <StageProgress current={done ? "done" : stage} progress={progress} />
          </div>

          {hasMedia && (
            <div className="relative mt-5 overflow-hidden rounded-2xl bg-surface-2 ring-1 ring-line-soft">
              {source.mediaType === "video" ? (
                <video
                  controls
                  preload="metadata"
                  className="max-h-72 w-full bg-ink"
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
        </div>
      </div>

      {error ? (
        <ErrorNotice error={error} title="Xử lý chưa hoàn tất" className="mt-4" />
      ) : null}

      {/* Voice on the left, the document taking shape on the right */}
      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        {/* LEFT: transcript as a timeline */}
        <section className="animate-enter flex max-h-[75dvh] min-h-[40dvh] flex-col overflow-hidden rounded-[var(--radius-bezel)] bg-surface shadow-[var(--shadow-card)] ring-1 ring-line/80 [animation-delay:140ms]">
          <header className="flex items-center justify-between border-b border-line-soft px-6 py-4">
            <p className="text-sm font-semibold text-ink">
              Lời thoại
              <span className="ml-2 font-mono text-xs font-normal tabular-nums text-ink-faint">
                {visibleSegments.length}
              </span>
            </p>
            <Waveform live={!done && !error} bars={16} className="h-4 w-24" />
          </header>
          <div ref={transcriptScroll} className="min-h-0 flex-1 overflow-y-auto px-6 py-6">
            <ol className="relative">
              {(() => {
                const blocks = groupSpeakerBlocks(visibleSegments);
                return blocks.map((block, b) => {
                  const startIndex = block.firstIndex;
                  const last = b === blocks.length - 1;
                  // Blocks streamed in live get the word cascade; replayed ones don't.
                  const live = startIndex >= snapshotCount;
                  const first = block.segments[0];
                  const color = getSpeakerStyle(block.speaker).color;
                  return (
                    <li
                      key={first.id ?? `${first.startMs}-${startIndex}`}
                      className={`relative pb-6 pl-7 ${live ? "" : "animate-rise"}`}
                      style={!live ? { animationDelay: `${Math.min(b * 30, 600)}ms` } : undefined}
                    >
                      {/* Timeline: rail and a dot in the speaker's color */}
                      {!last && (
                        <span aria-hidden="true" className="absolute bottom-0 left-[5px] top-4 w-px bg-line" />
                      )}
                      <span
                        aria-hidden="true"
                        className={`absolute left-0 top-[5px] h-[11px] w-[11px] rounded-full ring-4 ring-surface ${
                          last && !done ? "recording-dot" : ""
                        }`}
                        style={{ backgroundColor: color }}
                      />
                      <div className="mb-1.5 flex items-center gap-2.5">
                        <SpeakerChip speaker={block.speaker} />
                        {block.segments.some((seg) => seg.overlapped) && <CrosstalkBadge />}
                        <span className="font-mono text-[11px] text-ink-faint">
                          {formatStamp(block.startMs)}
                        </span>
                      </div>
                      <p
                        className={`text-[15px] leading-relaxed text-ink ${
                          last && !done ? "caret" : ""
                        }`}
                      >
                        {block.segments.map((seg, i) => (
                          <span
                            key={seg.id ?? `${seg.startMs}-${i}`}
                            className={
                              seg.overlapped
                                ? "underline decoration-dotted decoration-ink-faint underline-offset-4"
                                : undefined
                            }
                          >
                            {i > 0 && " "}
                            {live ? <RevealWords text={seg.text} /> : seg.text}
                          </span>
                        ))}
                      </p>
                    </li>
                  );
                });
              })()}
            </ol>
            {segments.length === 0 && !error && (
              <WaitingLines caption="Đang nghe đoạn đầu tiên…" />
            )}
            {segments.length > 0 && !done && !error && (
              <div className="flex items-center gap-2 pl-7 text-sm text-ink-faint" aria-live="polite">
                <span className="flex gap-1">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current [animation-delay:150ms]" />
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current [animation-delay:300ms]" />
                </span>
                Đang nhận dạng tiếp…
              </div>
            )}
          </div>
        </section>

        {/* RIGHT: the document, presented as a sheet of paper on a tray */}
        <section className="animate-enter flex max-h-[75dvh] min-h-[40dvh] flex-col overflow-hidden rounded-[var(--radius-bezel)] bg-surface-3/60 ring-1 ring-line/80 [animation-delay:200ms]">
          <header className="flex items-center justify-between px-6 py-4">
            <p className="text-sm font-semibold text-ink">
              Tài liệu
              <span className="ml-2 font-mono text-xs font-normal tabular-nums text-ink-faint">
                {sections.length} mục
              </span>
            </p>
            {!done && !error && sections.length > 0 && (
              <span className="flex items-center gap-1.5 text-xs font-medium text-signal-ink">
                <span className="recording-dot h-1.5 w-1.5 rounded-full bg-signal" />
                Đang viết
              </span>
            )}
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3 sm:px-4 sm:pb-4">
            <div className="min-h-full rounded-2xl bg-surface px-6 py-7 shadow-[var(--shadow-card)] sm:px-9 sm:py-9">
              <div className="space-y-8">
                {sections.map((sec, i) => {
                  const writing = !done && i === sections.length - 1;
                  return (
                    <article
                      key={sec.id}
                      className={`animate-unfold -mx-3 rounded-xl px-3 py-2 ${writing ? "ink-sweep" : ""}`}
                    >
                      <h2 className="heading-rule font-display text-xl font-semibold tracking-[-0.02em] text-ink">
                        {sec.heading}
                      </h2>
                      <div className="mt-3">
                        <MarkdownLite text={sec.markdown} live={writing} />
                      </div>
                    </article>
                  );
                })}
              </div>
              {sections.length === 0 && !error && (
                <WaitingLines heading caption="Tài liệu hình thành sau khi có lời thoại…" />
              )}
            </div>
          </div>
        </section>
      </div>

      <FinalizeDialog
        open={dialogOpen}
        liveDocumentId={liveDocumentId}
        onClose={() => setDialogOpen(false)}
        onOpen={() => setDialogOpen(true)}
      />
    </div>
  );
}

/** Splits a line into words that fade and sharpen in one after another. */
function RevealWords({ text }: { text: string }) {
  const words = text.split(/\s+/).filter(Boolean);
  return (
    <>
      {words.map((word, i) => (
        <span key={i}>
          {i > 0 && " "}
          <span
            className="word-in"
            // Cap the cascade so a long line still settles within ~1.2 s.
            style={{ animationDelay: `${Math.min(i * 35, 1200)}ms` }}
          >
            {word}
          </span>
        </span>
      ))}
    </>
  );
}

const STAGE_ORDER: Exclude<ProcessingStage, "done">[] = [
  "extracting_audio",
  "transcribing",
  "generating_doc",
];

const STAGE_COPY: Record<
  Exclude<ProcessingStage, "done">,
  { title: string; muted: string; icon: Icon }
> = {
  extracting_audio: {
    title: "Đang tách âm thanh.",
    muted: "Chuẩn bị cho phần nghe.",
    icon: WaveformIcon,
  },
  transcribing: {
    title: "Đang chép lời.",
    muted: "Từng câu hiện ra bên dưới.",
    icon: Quotes,
  },
  generating_doc: {
    title: "Đang dựng tài liệu.",
    muted: "Ý chính được sắp thành mục.",
    icon: MagicWand,
  },
};

/** A glowing core with sound rings; its icon follows the current stage. */
function StatusOrb({
  stage,
  done,
  failed,
}: {
  stage: ProcessingStage;
  done: boolean;
  failed: boolean;
}) {
  const live = !done && !failed;
  const StageIcon: Icon = failed
    ? WarningCircle
    : done
      ? Check
      : STAGE_COPY[stage === "done" ? "generating_doc" : stage].icon;
  const tone = failed ? "bg-danger" : done ? "bg-ok" : "bg-brand";

  return (
    <div className="relative flex h-16 w-16 shrink-0 items-center justify-center">
      {live &&
        [0, 0.8, 1.6].map((delay) => (
          <span
            key={delay}
            aria-hidden="true"
            className="orb-ring absolute inset-0 rounded-full border border-brand/50"
            style={{ animationDelay: `${delay}s` }}
          />
        ))}
      <span
        className={`relative flex h-14 w-14 items-center justify-center rounded-full text-on-brand shadow-[inset_0_1px_0_rgb(255_255_255/0.3),0_12px_28px_-10px_rgb(224_85_47/0.7)] transition-colors duration-700 ${tone}`}
      >
        <StageIcon
          key={failed ? "failed" : done ? "done" : stage}
          size={26}
          weight="fill"
          className="animate-pop"
        />
      </span>
    </div>
  );
}

/** Placeholder lines that shimmer in sequence while the first text arrives. */
function WaitingLines({ caption, heading = false }: { caption: string; heading?: boolean }) {
  const widths = heading
    ? ["w-2/5", "w-full", "w-11/12", "w-4/5", "w-3/5"]
    : ["w-1/4", "w-full", "w-5/6", "w-1/4", "w-11/12", "w-2/3"];
  return (
    <div aria-live="polite">
      <div className="space-y-3">
        {widths.map((w, i) => (
          <div
            key={i}
            className={`skeleton rounded-full ${w} ${
              heading && i === 0 ? "h-5" : i % 3 === 0 && !heading ? "h-3" : "h-3.5"
            }`}
            style={{ animationDelay: `${i * 140}ms` }}
          />
        ))}
      </div>
      <p className="mt-6 text-sm text-ink-faint">{caption}</p>
    </div>
  );
}
