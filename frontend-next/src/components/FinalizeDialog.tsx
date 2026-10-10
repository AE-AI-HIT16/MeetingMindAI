"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { finalizeDocument } from "@/lib/api";
import { useSession } from "next-auth/react";
import type { DocMode, DocumentGenerationStage } from "@/lib/types";
import { useDocumentGenerationEvents } from "@/lib/useDocumentGenerationEvents";
import { ErrorNotice } from "@/components/ErrorNotice";
import { Check, ListBullets, TextAlignJustify, WarningCircle } from "@phosphor-icons/react";

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
    icon: <ListBullets size={20} />,
  },
  {
    mode: "full_text",
    title: "Toàn văn",
    desc: "Giữ nguyên toàn bộ nội dung đã tổng hợp, chỉ chỉnh cho gọn gàng.",
    icon: <TextAlignJustify size={20} />,
  },
];

export function FinalizeDialog({
  open,
  liveDocumentId,
  onClose,
  onOpen,
}: {
  open: boolean;
  liveDocumentId: string | null;
  onClose: () => void;
  /** Reopens the dialog from the background progress pill. */
  onOpen: () => void;
}) {
  const router = useRouter();
  const { data: authSession } = useSession();
  const [submitting, setSubmitting] = useState<DocMode | null>(null);
  const [error, setError] = useState<unknown>(null);
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

  const readyHref =
    generation.status === "done" && generation.documentId && generationSourceId
      ? `/sources/${generationSourceId}?view=doc&documentId=${encodeURIComponent(generation.documentId)}`
      : null;
  const chosen = CHOICES.find((c) => c.mode === submitting);

  // Open the result automatically only while the dialog is on screen. If the
  // user hid it, the pill offers the result instead of yanking the page.
  useEffect(() => {
    if (open && readyHref) router.push(readyHref);
  }, [open, readyHref, router]);

  if (!open) {
    // Generation needs this page's connection to keep running on the server,
    // so it continues while the dialog is hidden and says so.
    if (!submitting) return null;
    return (
      <div className="animate-enter fixed bottom-5 right-5 z-40 w-[min(22rem,calc(100vw-2.5rem))]" aria-live="polite">
        <div className="bezel bg-white/40 backdrop-blur-xl">
          <div className="bezel-core flex items-center gap-3 p-3 pr-3.5">
            <span className={`relative flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-on-brand ${readyHref ? "bg-ok" : displayError ? "bg-danger" : "bg-brand"}`}>
              {controlsLocked && (
                <span aria-hidden="true" className="orb-ring absolute inset-0 rounded-full border border-brand/50" />
              )}
              {readyHref ? <Check size={18} weight="bold" /> : displayError ? <WarningCircle size={18} weight="bold" /> : chosen?.icon}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold text-ink">
                {readyHref
                  ? `${chosen?.title ?? "Tài liệu"} đã sẵn sàng`
                  : displayError
                    ? "Chưa tạo được tài liệu"
                    : `Đang tạo ${chosen?.title.toLowerCase() ?? "tài liệu"} · ${Math.round(generation.progress * 100)}%`}
              </p>
              <p className="truncate text-xs text-ink-faint">
                {readyHref ? "Bấm để mở tài liệu." : displayError ? "Mở để xem chi tiết." : "Giữ trang này mở cho tới khi xong."}
              </p>
            </div>
            {readyHref ? (
              <button type="button" onClick={() => router.push(readyHref)} className="btn btn-primary px-4 py-2">
                Mở
              </button>
            ) : (
              <button type="button" onClick={onOpen} className="btn btn-secondary px-4 py-2">
                Xem
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  async function choose(mode: Exclude<DocMode, "live">) {
    if (!liveDocumentId || controlsLocked) return;
    setSubmitting(mode);
    setError(null);
    setGenerationJobId(null);
    try {
      const accepted = await finalizeDocument(liveDocumentId, mode, authSession?.accessToken);
      if (accepted.status === "done") {
        router.push(
          `/sources/${accepted.sourceId}?view=doc&documentId=${encodeURIComponent(accepted.documentId)}`,
        );
        return;
      }
      setGenerationSourceId(accepted.sourceId);
      setGenerationJobId(accepted.generationJobId);
    } catch (finalizeError) {
      setError(finalizeError);
      setSubmitting(null);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/25 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="finalize-title"
        className="animate-enter bezel w-full max-w-xl bg-white/40 backdrop-blur-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="bezel-core p-7 sm:p-8">
          <p className="eyebrow">Xử lý xong</p>
          <h2 id="finalize-title" className="font-display mt-2 text-3xl font-semibold tracking-[-0.03em] text-ink">
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
                className={`group flex flex-col items-start rounded-2xl p-5 text-left ring-1 transition duration-300 hover:-translate-y-0.5 disabled:cursor-not-allowed ${
                  submitting === c.mode
                    ? "bg-brand-wash ring-brand/40"
                    : "bg-surface-2 ring-line-soft hover:bg-surface hover:shadow-[var(--shadow-card)] hover:ring-line disabled:opacity-50"
                }`}
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-wash text-brand transition group-hover:bg-brand group-hover:text-on-brand">
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
                  className="shimmer h-full rounded-full bg-brand transition-[width] duration-300"
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
            <ErrorNotice
              compact
              title="Tài liệu chưa sẵn sàng"
              error="Hệ thống vẫn đang xử lý lời thoại. Vui lòng đợi xử lý xong rồi chọn đầu ra."
              className="mt-4"
            />
          )}
          {displayError ? (
            <ErrorNotice
              compact
              error={displayError}
              title="Chưa tạo được tài liệu"
              className="mt-4"
            />
          ) : null}

          <button
            onClick={onClose}
            className="mt-6 w-full rounded-full py-2 text-center text-sm font-medium text-ink-soft transition hover:bg-surface-2 hover:text-ink"
          >
            {controlsLocked ? "Ẩn đi, tạo tiếp ở nền" : "Để sau"}
          </button>
        </div>
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
