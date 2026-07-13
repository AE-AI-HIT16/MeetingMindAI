"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { PageHeader, Waveform } from "@/components/ui";

const ACCEPT = "video/*,audio/*";

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);

  function onFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setPicked(files[0].name);
    // In production: upload → create job → router.push(`/sources/${id}`).
    // Base build: navigate to the mock processing source after a beat.
    setTimeout(() => router.push("/sources/live-1"), 900);
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-10 md:px-10">
      <PageHeader eyebrow="Bước 1 / 3 · Tải lên" title="Chọn media để tạo tài liệu" />

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          onFiles(e.dataTransfer.files);
        }}
        className={`mt-8 flex w-full flex-col items-center justify-center rounded-[var(--radius-card)] border-2 border-dashed px-6 py-16 text-center transition ${
          dragging
            ? "border-brand bg-brand-wash"
            : "border-line bg-surface hover:border-brand/60 hover:bg-surface-2"
        }`}
      >
        <Waveform
          live={dragging || !!picked}
          bars={40}
          className="mb-8 h-16 w-56"
        />

        {picked ? (
          <>
            <p className="font-display text-xl font-medium text-ink">
              Đang mở {picked}…
            </p>
            <p className="mt-2 text-sm text-ink-soft">
              Chuyển tới màn hình xử lý thời gian thực.
            </p>
          </>
        ) : (
          <>
            <p className="font-display text-xl font-medium text-ink">
              Kéo thả tệp vào đây
            </p>
            <p className="mt-2 text-sm text-ink-soft">
              hoặc <span className="font-medium text-brand">chọn từ máy</span>.
              Hỗ trợ MP4, MKV, WEBM, WAV, MP3, M4A — tối đa 2&nbsp;GB.
            </p>
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => onFiles(e.target.files)}
        />
      </button>

      {/* What happens next — set expectations, in the user's terms */}
      <ol className="mt-8 grid gap-3 sm:grid-cols-3">
        {[
          ["Tách & nghe", "Tách audio từ video, xử lý theo từng đoạn."],
          ["Hiện dần", "Lời thoại và tài liệu hình thành ngay khi xử lý."],
          ["Xuất bản", "Chọn tóm tắt hoặc toàn văn, rồi tải PDF / DOCX."],
        ].map(([title, desc], i) => (
          <li
            key={title}
            className="rounded-xl border border-line bg-surface p-4"
          >
            <span className="font-mono text-xs font-bold text-brand">
              0{i + 1}
            </span>
            <p className="mt-1.5 font-medium text-ink">{title}</p>
            <p className="mt-1 text-xs leading-relaxed text-ink-soft">{desc}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
