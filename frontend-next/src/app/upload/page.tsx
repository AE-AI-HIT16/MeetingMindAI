"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  FileArrowDown,
  TextAlignLeft,
  UploadSimple,
  Waveform as WaveformIcon,
} from "@phosphor-icons/react";
import { PageTitle, Waveform } from "@/components/ui";
import { uploadSource, warmupAsr } from "@/lib/api";
import { ErrorNotice } from "@/components/ErrorNotice";
import { EMPTY_HINTS, RecognitionHints } from "@/components/RecognitionHints";

const ACCEPT = "video/*,audio/*";
const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [hints, setHints] = useState(EMPTY_HINTS);

  // Start the RunPod cold start now; by the time the file is uploaded the
  // worker is (nearly) ready.
  useEffect(() => {
    void warmupAsr();
  }, []);

  async function onFiles(files: FileList | null) {
    if (!files || files.length === 0 || uploading) return;

    const file = files[0];
    if (file.size > MAX_UPLOAD_BYTES) {
      setPicked(null);
      setUploadProgress(0);
      setError("Tệp vượt quá giới hạn 2 GB.");
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    setPicked(file.name);
    setUploadProgress(0);
    setUploading(true);
    setError(null);

    // A long upload can outlast the worker idle timeout after the page-open
    // warmup: cold-start again near the end so processing starts warm.
    let warmedNearEnd = false;
    const onProgress = (percent: number) => {
      setUploadProgress(percent);
      if (!warmedNearEnd && percent >= 80) {
        warmedNearEnd = true;
        void warmupAsr();
      }
    };

    try {
      const result = await uploadSource(file, onProgress, {
        context: hints.context,
      });
      router.push(
        `/sources/${result.sourceId}?jobId=${encodeURIComponent(result.jobId)}`,
      );
    } catch (uploadError) {
      setError(uploadError);
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 md:px-10 md:py-12">
      <div className="animate-enter">
        <PageTitle title="Tải file lên" lede="Thả file vào, máy bắt đầu nghe ngay." />
      </div>

      <div className="mt-10 grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {/* Drop zone */}
        <div className="bezel animate-enter [animation-delay:60ms]">
          <button
            type="button"
            disabled={uploading}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              void onFiles(e.dataTransfer.files);
            }}
            className={`relative flex min-h-[380px] w-full flex-col items-center justify-center overflow-hidden rounded-[calc(var(--radius-bezel)-6px)] border-2 border-dashed px-6 py-14 text-center shadow-[var(--shadow-card)] transition disabled:cursor-wait ${
              dragging
                ? "border-brand bg-brand-wash"
                : "border-line bg-surface hover:border-brand/50"
            }`}
          >
            <span
              aria-hidden="true"
              className={`pointer-events-none absolute left-1/2 top-1/3 h-72 w-72 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand/10 blur-3xl transition-opacity duration-500 ${
                dragging || picked ? "opacity-100" : "opacity-40"
              }`}
            />
            <Waveform
              live={dragging || !!picked}
              bars={48}
              className="relative mb-10 h-20 w-72 max-w-full"
            />

            {uploading && picked ? (
              <div className="relative flex w-full flex-col items-center">
                {/* Same live orb as the processing screen, so the hand-off feels continuous */}
                <div className="relative mb-6 flex h-14 w-14 items-center justify-center">
                  {[0, 0.8, 1.6].map((delay) => (
                    <span
                      key={delay}
                      aria-hidden="true"
                      className="orb-ring absolute inset-0 rounded-full border border-brand/50"
                      style={{ animationDelay: `${delay}s` }}
                    />
                  ))}
                  <span className="relative flex h-12 w-12 items-center justify-center rounded-full bg-brand text-on-brand shadow-[inset_0_1px_0_rgb(255_255_255/0.3)]">
                    <UploadSimple size={22} weight="bold" />
                  </span>
                </div>
                <p className="max-w-full truncate font-display text-xl font-semibold text-ink">
                  Đang tải {picked}
                </p>
                <div
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={uploadProgress}
                  aria-label="Tiến độ tải lên"
                  className="mt-5 h-1.5 w-full max-w-sm overflow-hidden rounded-full bg-line"
                >
                  <div
                    className="shimmer h-full rounded-full bg-brand transition-[width] duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="mt-3 font-mono text-2xl tabular-nums text-ink">
                  {uploadProgress}%
                </p>
              </div>
            ) : (
              <div className="relative flex flex-col items-center">
                <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand text-on-brand">
                  <UploadSimple size={22} weight="bold" />
                </span>
                <p className="mt-5 font-display text-2xl font-semibold tracking-tight text-ink">
                  Kéo thả file vào đây
                </p>
                <p className="mt-2 text-sm text-ink-soft">
                  hoặc <span className="font-medium text-brand-ink underline decoration-brand/40 underline-offset-4">chọn từ máy</span>
                </p>
                <p className="mt-6 font-mono text-xs text-ink-faint">
                  MP4 MKV WEBM WAV MP3 M4A, tối đa 2&nbsp;GB
                </p>
              </div>
            )}

            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              className="hidden"
              disabled={uploading}
              onChange={(e) => void onFiles(e.target.files)}
            />
          </button>
        </div>

        {/* Side column: optional hints, then what happens next */}
        <div className="animate-enter flex flex-col gap-6 [animation-delay:120ms]">
          <div className="rounded-[var(--radius-bezel)] bg-surface p-6 shadow-[var(--shadow-card)] ring-1 ring-line/80">
            <RecognitionHints
              value={hints}
              onChange={setHints}
              disabled={uploading}
            />
          </div>

          <ol className="relative space-y-6 pl-1">
            {STEPS.map(({ icon: StepIcon, title, desc }, i) => (
              <li key={title} className="relative flex gap-4">
                {i < STEPS.length - 1 && (
                  <span aria-hidden="true" className="absolute left-[17px] top-10 h-[calc(100%-12px)] w-px bg-line" />
                )}
                <span className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-surface-2 text-brand ring-1 ring-line">
                  <StepIcon size={18} />
                </span>
                <div className="pt-1">
                  <p className="text-sm font-medium text-ink">{title}</p>
                  <p className="mt-0.5 text-sm leading-relaxed text-ink-faint">{desc}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </div>

      {error ? (
        <ErrorNotice
          error={error}
          title="Chưa tải lên được"
          onDismiss={() => setError(null)}
          className="mt-6"
        />
      ) : null}
    </div>
  );
}

const STEPS = [
  {
    icon: WaveformIcon,
    title: "Tách và nghe",
    desc: "Tách audio từ video, nhận dạng lời nói theo từng đoạn.",
  },
  {
    icon: TextAlignLeft,
    title: "Hiện dần",
    desc: "Lời thoại và tài liệu hình thành ngay trong lúc xử lý.",
  },
  {
    icon: FileArrowDown,
    title: "Xuất bản",
    desc: "Chọn tóm tắt hoặc toàn văn, rồi tải PDF hay DOCX.",
  },
];
