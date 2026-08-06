"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { PageHeader, Waveform } from "@/components/ui";
import { uploadSource } from "@/lib/api";

const ACCEPT = "video/*,audio/*";
const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024;

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

    try {
      const result = await uploadSource(file, setUploadProgress);
      router.push(
        `/sources/${result.sourceId}?jobId=${encodeURIComponent(result.jobId)}`,
      );
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Không thể tải file lên.",
      );
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-10 md:px-10">
      <PageHeader eyebrow="Bước 1 / 3 · Tải lên" title="Chọn media để tạo tài liệu" />

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
        className={`mt-8 flex w-full flex-col items-center justify-center rounded-[var(--radius-card)] border-2 border-dashed px-6 py-16 text-center transition disabled:cursor-wait ${
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

        {uploading && picked ? (
          <>
            <p className="font-display text-xl font-medium text-ink">
              Đang tải {picked}… {uploadProgress}%
            </p>
            <p className="mt-2 text-sm text-ink-soft">
              Source và Job sẽ được tạo sau khi tải lên hoàn tất.
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
          disabled={uploading}
          onChange={(e) => void onFiles(e.target.files)}
        />
      </button>

      {error && (
        <p
          role="alert"
          className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {error}
        </p>
      )}

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
