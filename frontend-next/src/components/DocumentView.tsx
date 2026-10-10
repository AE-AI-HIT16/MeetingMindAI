"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import type {
  DocxExportPreset,
  DocumentData,
  ExportFormat,
  PdfExportPreset,
  Source,
} from "@/lib/types";
import { fetchDocumentExport, mediaUrl, saveBlob } from "@/lib/api";
import { ExportPreview } from "@/components/ExportPreview";
import { useSession } from "next-auth/react";
import { EditableSegmentText } from "@/components/EditableSegmentText";
import { formatDuration, formatStamp } from "@/lib/format";
import { useJobEvents } from "@/lib/useJobEvents";
import { MarkdownLite, headingId } from "@/components/MarkdownLite";
import { DocumentInsights } from "@/components/DocumentInsights";
import { CrosstalkBadge, DocModeBadge, SpeakerChip } from "@/components/ui";
import { groupSpeakerBlocks } from "@/lib/speakerBlocks";
import { ErrorNotice } from "@/components/ErrorNotice";
import {
  ArrowLeft,
  ArrowsIn,
  ArrowsOut,
  Check,
  CircleNotch,
  DownloadSimple,
  Eye,
  FileText,
  PencilSimple,
  Quotes,
} from "@phosphor-icons/react";

type Tab = "doc" | "transcript";

const EXPORT_FORMATS: { value: ExportFormat; label: string }[] = [
  { value: "pdf", label: "PDF" },
  { value: "docx", label: "DOCX" },
  { value: "md", label: "Markdown" },
];

const PDF_PRESETS: {
  value: PdfExportPreset;
  label: string;
  description: string;
}[] = [
  {
    value: "minimal",
    label: "Tối giản",
    description: "Sạch và tập trung vào nội dung",
  },
  {
    value: "blue_modern",
    label: "Xanh hiện đại",
    description: "Bố cục báo cáo mang màu MeetingMind",
  },
];

const DOCX_PRESETS: {
  value: DocxExportPreset;
  label: string;
  description: string;
}[] = [
  {
    value: "minimal",
    label: "Tối giản",
    description: "Tài liệu Word cơ bản, dễ chỉnh sửa",
  },
  {
    value: "modern",
    label: "Hiện đại",
    description: "Trang bìa kèm ngày tạo và tác giả",
  },
];

function segmentKey(id: number | null, startMs: number) {
  return id === null ? `start-${startMs}` : `id-${id}`;
}

export function DocumentView({
  source,
  document,
}: {
  source: Source;
  document: DocumentData;
}) {
  const [tab, setTab] = useState<Tab>("doc");
  // Video plays in a compact strip; this lets the viewer enlarge it in place.
  const [videoLarge, setVideoLarge] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [busy, setBusy] = useState<"preview" | "download" | null>(null);
  const [preview, setPreview] = useState<{
    file: { blob: Blob; filename: string };
    format: ExportFormat;
    presetLabel?: string;
  } | null>(null);
  const [downloadError, setDownloadError] = useState<unknown>(null);
  const { data: authSession } = useSession();
  const [exportFormat, setExportFormat] = useState<ExportFormat>("pdf");
  const [pdfPreset, setPdfPreset] =
    useState<PdfExportPreset>("minimal");
  const [docxPreset, setDocxPreset] =
    useState<DocxExportPreset>("minimal");
  const [playbackMs, setPlaybackMs] = useState<number | null>(null);
  // Speaker block being edited (one sentence per line) and saved edits.
  const [editingBlock, setEditingBlock] = useState<string | null>(null);
  const [editedText, setEditedText] = useState<Record<string, string>>({});
  const audioRef = useRef<HTMLAudioElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const segmentRefs = useRef(new Map<string, HTMLElement>());
  const { segments } = useJobEvents(source.jobId);

  const selectedPreset =
    exportFormat === "pdf"
      ? pdfPreset
      : exportFormat === "docx"
        ? docxPreset
        : undefined;
  const exportPresets =
    exportFormat === "pdf"
      ? PDF_PRESETS
      : exportFormat === "docx"
        ? DOCX_PRESETS
        : [];

  async function runExport(kind: "preview" | "download") {
    setBusy(kind);
    setDownloadError(null);
    try {
      const file = await fetchDocumentExport(
        document.id,
        exportFormat,
        selectedPreset,
        authSession?.accessToken,
      );
      if (kind === "preview") {
        setPreview({
          file,
          format: exportFormat,
          presetLabel: exportPresets.find((p) => p.value === selectedPreset)?.label,
        });
      } else {
        saveBlob(file.blob, file.filename);
        setExportOpen(false);
      }
    } catch (err) {
      setDownloadError(err);
    } finally {
      setBusy(null);
    }
  }

  const activeSegmentKey = useMemo(() => {
    if (playbackMs === null) return null;

    const segmentIndex = segments.findLastIndex(
      (segment) => segment.startMs <= playbackMs,
    );
    if (segmentIndex === -1) return null;

    const segment = segments[segmentIndex];
    const nextSegment = segments[segmentIndex + 1];
    const activeUntilMs = nextSegment
      ? Math.max(segment.endMs, nextSegment.startMs)
      : segment.endMs;

    return playbackMs <= activeUntilMs
      ? segmentKey(segment.id, segment.startMs)
      : null;
  }, [playbackMs, segments]);

  // Sections for the table of contents, from the "## " headings.
  const toc = useMemo(
    () =>
      document.markdown
        .split("\n")
        .filter((line) => line.startsWith("## "))
        .map((line) => {
          const title = line.slice(3).replace(/[*_`]/g, "").trim();
          return { title, id: headingId(title) };
        }),
    [document.markdown],
  );
  const [activeSection, setActiveSection] = useState<string | null>(null);
  // Full text: bring one speaker's turns forward (null speaker = unknown).
  const [focus, setFocus] = useState<{ speaker: number | null } | null>(null);

  useEffect(() => {
    if (tab !== "doc" || toc.length < 2) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) setActiveSection(visible[0].target.id);
      },
      { rootMargin: "-160px 0px -65% 0px" },
    );
    toc.forEach(({ id }) => {
      const el = window.document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [tab, toc]);

  const activeSpeaker = useMemo(() => {
    if (activeSegmentKey === null) return null;
    const segment = segments.find(
      (s) => segmentKey(s.id, s.startMs) === activeSegmentKey,
    );
    return segment ? segment.speaker : null;
  }, [activeSegmentKey, segments]);

  useEffect(() => {
    if (tab !== "transcript" || activeSegmentKey === null) return;

    const frame = window.requestAnimationFrame(() => {
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      segmentRefs.current.get(activeSegmentKey)?.scrollIntoView({
        behavior: reduceMotion ? "auto" : "smooth",
        block: "center",
      });
    });

    return () => window.cancelAnimationFrame(frame);
  }, [activeSegmentKey, tab]);

  function syncPlaybackTime(
    event: React.SyntheticEvent<HTMLMediaElement>,
  ) {
    setPlaybackMs(event.currentTarget.currentTime * 1000);
  }

  function seekTo(startMs: number) {
    const media =
      source.mediaType === "video" ? videoRef.current : audioRef.current;
    if (!media) return;
    media.currentTime = startMs / 1000;
    setPlaybackMs(startMs);
    void media.play().catch(() => {
      // Browser controls still allow manual playback if autoplay is blocked.
    });
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 md:px-10 md:py-10">
      <Link
        href="/"
        className="eyebrow inline-flex items-center gap-1.5 transition hover:text-brand-ink"
      >
        <ArrowLeft size={12} />
        Thư viện
      </Link>

      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink [overflow-wrap:anywhere]">
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
            <DocModeBadge mode={document.mode} />
          </p>
        </div>

        {/* Export */}
        <div className="relative z-30">
          <button
            onClick={() => setExportOpen((v) => !v)}
            className="btn btn-primary"
          >
            <DownloadSimple size={16} weight="bold" />
            Xuất
          </button>
          {exportOpen && (
            <div className="animate-enter absolute left-0 z-30 mt-3 w-[26rem] max-w-[calc(100vw-2rem)] sm:left-auto sm:right-0">
              <div className="bezel bg-white/50 backdrop-blur-xl">
                <div className="bezel-core p-5">
                  <fieldset>
                    <legend className="text-sm font-semibold text-ink">Định dạng</legend>
                    <div className="mt-2 grid grid-cols-3 gap-1 rounded-full bg-surface-3/70 p-1 ring-1 ring-line-soft">
                      {EXPORT_FORMATS.map(({ value, label }) => (
                        <button
                          key={value}
                          type="button"
                          aria-pressed={exportFormat === value}
                          onClick={() => setExportFormat(value)}
                          className={`rounded-full px-2 py-1.5 text-xs font-semibold transition ${
                            exportFormat === value
                              ? "bg-surface text-ink shadow-[var(--shadow-card)]"
                              : "text-ink-faint hover:text-ink"
                          }`}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </fieldset>

                  {exportFormat !== "md" ? (
                    <fieldset className="mt-5">
                      <legend className="text-sm font-semibold text-ink">Mẫu trình bày</legend>
                      {/* Thumbnails are page 1 of each real template, rendered
                          from the export endpoint (public/export-presets). */}
                      <div className="mt-2 grid grid-cols-2 gap-3">
                        {exportPresets.map((preset) => {
                          const selected = selectedPreset === preset.value;
                          return (
                            <button
                              key={preset.value}
                              type="button"
                              aria-pressed={selected}
                              onClick={() => {
                                if (exportFormat === "pdf") {
                                  setPdfPreset(preset.value as PdfExportPreset);
                                } else {
                                  setDocxPreset(preset.value as DocxExportPreset);
                                }
                              }}
                              className="group text-left"
                            >
                              <span
                                className={`relative block overflow-hidden rounded-xl bg-surface ring-2 transition duration-300 ${
                                  selected
                                    ? "ring-brand shadow-[0_12px_28px_-14px_rgb(224_85_47/0.6)]"
                                    : "ring-line group-hover:ring-ink-faint/50"
                                }`}
                              >
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                  src={`/export-presets/${exportFormat}-${preset.value}.webp`}
                                  alt={`Trang đầu mẫu ${preset.label}`}
                                  width={320}
                                  height={413}
                                  className="block aspect-[320/413] w-full object-cover object-top transition duration-500 group-hover:scale-[1.03]"
                                />
                                {selected && (
                                  <span className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-brand text-on-brand shadow">
                                    <Check size={13} weight="bold" />
                                  </span>
                                )}
                              </span>
                              <span className="mt-2 block text-sm font-semibold text-ink">{preset.label}</span>
                              <span className="block text-xs leading-snug text-ink-faint">{preset.description}</span>
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>
                  ) : (
                    <p className="mt-5 rounded-xl bg-surface-2 px-3 py-3 text-xs leading-relaxed text-ink-soft">
                      Tải nội dung Markdown nguyên bản để tiếp tục chỉnh sửa.
                    </p>
                  )}

                  <div className="mt-5 grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      disabled={busy !== null}
                      onClick={() => void runExport("preview")}
                      className="btn btn-secondary"
                    >
                      {busy === "preview" ? <CircleNotch size={16} className="animate-spin" /> : <Eye size={16} />}
                      Xem trước
                    </button>
                    <button
                      type="button"
                      disabled={busy !== null}
                      onClick={() => void runExport("download")}
                      className="btn btn-primary"
                    >
                      {busy === "download" ? <CircleNotch size={16} className="animate-spin" /> : <DownloadSimple size={16} weight="bold" />}
                      Tải .{exportFormat}
                    </button>
                  </div>
                  {downloadError ? (
                    <ErrorNotice
                      compact
                      error={downloadError}
                      title="Chưa tạo được file"
                      className="mt-3"
                    />
                  ) : null}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tabs: a segmented control */}
      <div
        role="tablist"
        className="mt-8 inline-flex gap-1 rounded-full bg-surface-3/70 p-1 ring-1 ring-line-soft"
      >
        {(
          [
            ["doc", "Tài liệu", FileText],
            ["transcript", "Lời thoại", Quotes],
          ] as const
        ).map(([key, label, TabIcon]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
            className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition ${
              tab === key
                ? "bg-surface text-ink shadow-[var(--shadow-card)]"
                : "text-ink-faint hover:text-ink"
            }`}
          >
            <TabIcon size={16} weight={tab === key ? "fill" : "regular"} />
            {label}
          </button>
        ))}
      </div>

      {/* One media element, sticky under both tabs so playback never stops.
          The header shows who is speaking at the current playback time. */}
      <section
        aria-label="Trình phát media"
        className="sticky top-3 z-20 mt-5 rounded-[var(--radius-bezel)] bg-surface/85 p-3 shadow-[var(--shadow-lift)] ring-1 ring-line/80 backdrop-blur-xl"
      >
        <div className="mb-2 flex min-h-7 items-center justify-between gap-3 px-1">
          <span className="flex min-w-0 items-center gap-2 text-xs font-medium text-ink-soft">
            {playbackMs !== null && activeSpeaker !== null ? (
              <>
                <span className="recording-dot h-1.5 w-1.5 shrink-0 rounded-full bg-signal" />
                Đang nói
                <SpeakerChip speaker={activeSpeaker} />
              </>
            ) : (
              <>
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-line" />
                Bấm vào một câu để nghe lại đúng đoạn đó
              </>
            )}
          </span>
          {source.mediaType === "video" && (
            <button
              type="button"
              onClick={() => setVideoLarge((v) => !v)}
              className="flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium text-ink-soft transition hover:bg-surface-3 hover:text-ink"
            >
              {videoLarge ? <ArrowsIn size={14} /> : <ArrowsOut size={14} />}
              {videoLarge ? "Thu nhỏ" : "Phóng to"}
            </button>
          )}
        </div>

        <div className="overflow-hidden rounded-[calc(var(--radius-bezel)-12px)] bg-surface-2">
          {source.mediaType === "video" ? (
            <video
              ref={videoRef}
              controls
              preload="metadata"
              onLoadedMetadata={syncPlaybackTime}
              onPlay={syncPlaybackTime}
              onSeeked={syncPlaybackTime}
              onTimeUpdate={syncPlaybackTime}
              className={`w-full bg-ink transition-[max-height] duration-500 ${
                videoLarge ? "max-h-[70dvh]" : "max-h-48"
              }`}
              src={mediaUrl(source.id)}
            />
          ) : (
            <audio
              ref={audioRef}
              controls
              preload="metadata"
              onLoadedMetadata={syncPlaybackTime}
              onPlay={syncPlaybackTime}
              onSeeked={syncPlaybackTime}
              onTimeUpdate={syncPlaybackTime}
              className="w-full"
              src={mediaUrl(source.id)}
            />
          )}
        </div>
      </section>

      {/* Document and transcript remain readable behind the sticky player. */}
      <div className="py-8">
        {tab === "doc" && (
          <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_15rem]">
            {/* The document as a sheet of paper on a tray */}
            <div className="bezel animate-enter">
              <article className="bezel-core px-6 py-10 sm:px-12 sm:py-14 lg:px-16">
                <MarkdownLite
                  text={document.markdown}
                  variant="document"
                  onSeek={seekTo}
                  focus={focus}
                />
              </article>
            </div>

            {/* Side panel: meeting facts, who spoke how much, and the contents */}
            <aside className="xl:order-last">
              <div className="space-y-8 xl:sticky xl:top-40 xl:max-h-[calc(100dvh-11rem)] xl:overflow-y-auto xl:pb-6">
                <DocumentInsights
                  segments={segments}
                  durationMs={source.durationMs}
                  focus={focus}
                  onFocus={document.mode === "full_text" ? setFocus : undefined}
                />

                {toc.length > 1 && (
                  <nav aria-label="Mục lục">
                    <p className="text-xs font-semibold text-ink-faint">Mục lục</p>
                    <ol className="mt-3 space-y-0.5 border-l border-line">
                      {toc.map((item, i) => (
                        <li key={item.id}>
                          <a
                            href={`#${item.id}`}
                            className={`-ml-px flex gap-2.5 border-l-2 py-1.5 pl-3 text-[13px] leading-snug transition ${
                              activeSection === item.id
                                ? "border-brand font-medium text-ink"
                                : "border-transparent text-ink-faint hover:text-ink"
                            }`}
                          >
                            <span className="font-mono text-[11px] text-brand/80">
                              {String(i + 1).padStart(2, "0")}
                            </span>
                            <span className="line-clamp-2">{item.title}</span>
                          </a>
                        </li>
                      ))}
                    </ol>
                  </nav>
                )}
              </div>
            </aside>
          </div>
        )}

        {tab === "transcript" && (
          <div className="space-y-2">
            {groupSpeakerBlocks(segments).map((block) => {
              const keys = block.segments.map((seg) => segmentKey(seg.id, seg.startMs));
              const blockActive = activeSegmentKey !== null && keys.includes(activeSegmentKey);
              return (
                <div
                  key={keys[0]}
                  className={`relative flex gap-4 rounded-2xl border px-3 py-3 transition-all duration-300 ${
                    blockActive
                      ? "border-brand/25 bg-brand-wash shadow-[0_12px_32px_-18px_rgb(44_62_224_/_0.55)]"
                      : activeSegmentKey
                        ? "border-transparent opacity-55"
                        : "border-transparent"
                  }`}
                >
                  {blockActive && (
                    <span
                      aria-hidden="true"
                      className="absolute inset-y-3 left-0 w-1 rounded-r-full bg-brand"
                    />
                  )}
                  <button
                    type="button"
                    onClick={() => seekTo(block.startMs)}
                    title={`Phát từ ${formatStamp(block.startMs)}`}
                    className={`w-12 shrink-0 self-start pt-0.5 text-left font-mono text-[11px] transition ${
                      blockActive
                        ? "font-semibold text-brand-ink"
                        : "text-ink-faint hover:text-brand"
                    }`}
                  >
                    {formatStamp(block.startMs)}
                  </button>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <SpeakerChip speaker={block.speaker} />
                      {block.segments.some((seg) => seg.overlapped) && <CrosstalkBadge />}
                      {blockActive && (
                        <span className="inline-flex items-center gap-1.5 rounded-full bg-brand px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-on-brand">
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-on-brand" />
                          Đang phát
                        </span>
                      )}
                      {source.jobId && (
                        <button
                          type="button"
                          onClick={() =>
                            setEditingBlock(editingBlock === keys[0] ? null : keys[0])
                          }
                          aria-pressed={editingBlock === keys[0]}
                          className={`ml-auto inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs font-medium transition ${
                            editingBlock === keys[0]
                              ? "border-brand bg-brand text-on-brand"
                              : "border-line text-ink-soft hover:border-brand hover:bg-brand-wash hover:text-brand-ink"
                          }`}
                        >
                          <PencilSimple size={12} aria-hidden="true" />
                          {editingBlock === keys[0] ? "Xong" : "Sửa"}
                        </button>
                      )}
                    </div>
                    {editingBlock === keys[0] ? (
                      <div className="mt-2 space-y-2">
                        {block.segments.map((seg, index) => (
                          <div key={keys[index]} className="flex gap-3">
                            <button
                              type="button"
                              onClick={() => seekTo(seg.startMs)}
                              title={`Phát từ ${formatStamp(seg.startMs)}`}
                              className="w-10 shrink-0 self-start pt-1 text-left font-mono text-[11px] text-ink-faint hover:text-brand"
                            >
                              {formatStamp(seg.startMs)}
                            </button>
                            <EditableSegmentText
                              jobId={source.jobId}
                              segmentId={seg.id}
                              text={editedText[keys[index]] ?? seg.text}
                              onSaved={(text) =>
                                setEditedText((prev) => ({ ...prev, [keys[index]]: text }))
                              }
                              className="min-w-0 flex-1 text-[15px]"
                            />
                          </div>
                        ))}
                      </div>
                    ) : (
                    <div className="mt-1 text-[15px]">
                      {block.segments.map((seg, index) => {
                        const key = keys[index];
                        const isActive = key === activeSegmentKey;
                        return (
                          <span
                            key={key}
                            ref={(node) => {
                              if (node) {
                                segmentRefs.current.set(key, node);
                              } else {
                                segmentRefs.current.delete(key);
                              }
                            }}
                            aria-current={isActive ? "true" : undefined}
                          >
                            {index > 0 && " "}
                            <EditableSegmentText
                              inline
                              jobId={source.jobId}
                              segmentId={seg.id}
                              text={editedText[key] ?? seg.text}
                              onSaved={(text) =>
                                setEditedText((prev) => ({ ...prev, [key]: text }))
                              }
                              onClick={() => seekTo(seg.startMs)}
                              className={`transition-colors duration-300 ${
                                isActive ? "bg-brand/15 font-semibold" : "hover:bg-surface-2"
                              } ${
                                seg.overlapped
                                  ? "underline decoration-dotted decoration-ink-faint underline-offset-4"
                                  : ""
                              }`}
                            />
                          </span>
                        );
                      })}
                    </div>
                    )}
                  </div>
                </div>
              );
            })}
            {segments.length === 0 && (
              <p className="text-sm text-ink-faint">
                Đang tải lời thoại…
              </p>
            )}
          </div>
        )}
      </div>

      {preview && (
        <ExportPreview {...preview} onClose={() => setPreview(null)} />
      )}
    </div>
  );
}
