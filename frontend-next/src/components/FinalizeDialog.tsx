"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { finalizeDocument } from "@/lib/api";
import type { DocMode, DocumentGenerationStage } from "@/lib/types";
import { useDocumentGenerationEvents } from "@/lib/useDocumentGenerationEvents";

const CHOICES: {
  mode: Exclude<DocMode, "live">;
  title: string;
  desc: string;
  icon: React.ReactNode;
}[] = [
  {
    mode: "summary",
    title: "Tóm tắt",
    desc: "Rút gọn thành các chủ đề, quyết định và việc cần làm. Ngắn, dễ đọc.",
    icon: <SummaryIcon />,
  },
  {
    mode: "full_text",
    title: "Toàn văn",
    desc: "Giữ nguyên toàn bộ nội dung đã tổng hợp, chỉ chỉnh cho gọn gàng.",
    icon: <FullTextIcon />,
  },
];

export function FinalizeDialog({
  open,
  liveDocumentId,
  onClose,
}: {
  open: boolean;
  liveDocumentId: string | null;
  onClose: () => void;
}) {
  const router = useRouter();
  const [submitting, setSubmitting] = useState<DocMode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generationJobId, setGenerationJobId] = useState<string | null>(
    null,
  );
  const [generationSourceId, setGenerationSourceId] = useState<string | null>(
    null,
  );
  const generation = useDocumentGenerationEvents(generationJobId);
  const generationActive =
    generation.status === "queued" ||
    generation.status === "processing";
  const requestPending = submitting !== null && generationJobId === null;
  const controlsLocked = requestPending || generationActive;
  const displayError =
    error ??
    (generation.status === "failed" ? generation.error : null);

  useEffect(() => {
    if (
      generation.status === "done" &&
      generation.documentId &&
      generationSourceId
    ) {
      router.push(
        `/sources/${generationSourceId}?view=doc&documentId=${encodeURIComponent(generation.documentId)}`,
      );
      return;
    }
  }, [
    generation.documentId,
    generation.status,
    generationSourceId,
    router,
  ]);

  if (!open) return null;

  async function choose(mode: Exclude<DocMode, "live">) {
    if (!liveDocumentId || controlsLocked) return;
    setSubmitting(mode);
    setError(null);
    setGenerationJobId(null);
    try {
      const accepted = await finalizeDocument(liveDocumentId, mode);
      if (accepted.status === "done") {
        router.push(
          `/sources/${accepted.sourceId}?view=doc&documentId=${encodeURIComponent(accepted.documentId)}`,
        );
        return;
      }
      setGenerationSourceId(accepted.sourceId);
      setGenerationJobId(accepted.generationJobId);
    } catch (finalizeError) {
      setError(
        finalizeError instanceof Error
          ? finalizeError.message
          : "Không thể tạo tài liệu.",
      );
      setSubmitting(null);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4 backdrop-blur-sm"
      onClick={() => {
        if (!controlsLocked) onClose();
      }}
    >
      <div
        className="w-full max-w-lg rounded-[var(--radius-card)] border border-line bg-surface p-7 shadow-[var(--shadow-lift)]"
        onClick={(e) => e.stopPropagation()}
      >
        <p className="eyebrow">Xử lý xong · Bước 3 / 3</p>
        <h2 className="font-display mt-2 text-2xl font-semibold text-ink">
          Bạn muốn tài liệu dạng nào?
        </h2>
        <p className="mt-1.5 text-sm text-ink-soft">
          Có thể tạo dạng còn lại bất cứ lúc nào sau đó.
        </p>

        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          {CHOICES.map((c) => (
            <button
              key={c.mode}
              disabled={!liveDocumentId || controlsLocked}
              onClick={() => void choose(c.mode)}
              className="group flex flex-col items-start rounded-2xl border border-line bg-surface-2 p-5 text-left transition hover:-translate-y-0.5 hover:border-brand hover:bg-brand-wash/50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-wash text-brand transition group-hover:bg-brand group-hover:text-white">
                {c.icon}
              </span>
              <span className="mt-3 font-display text-lg font-medium text-ink">
                {c.title}
              </span>
              <span className="mt-1 text-xs leading-relaxed text-ink-soft">
                {controlsLocked && submitting === c.mode
                  ? generationStageLabel(generation.stage)
                  : c.desc}
              </span>
            </button>
          ))}
        </div>

        {controlsLocked && (
          <div className="mt-4" aria-live="polite">
            <div className="mb-1.5 flex items-center justify-between text-xs text-ink-soft">
              <span>{generationStageLabel(generation.stage)}</span>
              <span>{Math.round(generation.progress * 100)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-line">
              <div
                className="h-full rounded-full bg-brand transition-[width] duration-300"
                style={{
                  width: `${Math.max(
                    submitting && !generationJobId ? 3 : 0,
                    Math.round(generation.progress * 100),
                  )}%`,
                }}
              />
            </div>
            {generationJobId && !generation.isConnected && (
              <p className="mt-2 text-xs text-ink-faint">
                Đang kết nối lại với tiến trình trên server…
              </p>
            )}
          </div>
        )}

        {!liveDocumentId && (
          <p className="mt-4 text-sm text-red-700">
            Job chưa tạo xong tài liệu live để finalize.
          </p>
        )}
        {displayError && (
          <p role="alert" className="mt-4 text-sm text-red-700">
            {displayError}
          </p>
        )}

        <button
          disabled={controlsLocked}
          onClick={onClose}
          className="mt-5 w-full text-center text-sm text-ink-faint hover:text-ink disabled:opacity-50"
        >
          Để sau
        </button>
      </div>
    </div>
  );
}

function generationStageLabel(
  stage: DocumentGenerationStage | null,
): string {
  switch (stage) {
    case "queued":
      return "Đang chờ tạo tài liệu…";
    case "generating":
      return "Đang tạo tài liệu…";
    case "saving":
      return "Đang lưu tài liệu…";
    case "done":
      return "Đã tạo xong.";
    case "failed":
      return "Tạo tài liệu thất bại.";
    default:
      return "Đang gửi yêu cầu…";
  }
}

function SummaryIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M4 7h16M4 12h10M4 17h7" />
    </svg>
  );
}
function FullTextIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M5 4h14M5 8h14M5 12h14M5 16h14M5 20h10" />
    </svg>
  );
}
