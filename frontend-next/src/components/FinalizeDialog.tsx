"use client";

import { useRouter } from "next/navigation";
import type { DocMode } from "@/lib/types";

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
  sourceId,
  onClose,
}: {
  open: boolean;
  sourceId: string;
  onClose: () => void;
}) {
  const router = useRouter();
  if (!open) return null;

  function choose(mode: DocMode) {
    // Real build: POST /v1/documents/{id}/finalize { mode } then refresh.
    router.push(`/sources/${sourceId}?view=doc&mode=${mode}`);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4 backdrop-blur-sm"
      onClick={onClose}
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
              onClick={() => choose(c.mode)}
              className="group flex flex-col items-start rounded-2xl border border-line bg-surface-2 p-5 text-left transition hover:-translate-y-0.5 hover:border-brand hover:bg-brand-wash/50"
            >
              <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-wash text-brand transition group-hover:bg-brand group-hover:text-white">
                {c.icon}
              </span>
              <span className="mt-3 font-display text-lg font-medium text-ink">
                {c.title}
              </span>
              <span className="mt-1 text-xs leading-relaxed text-ink-soft">
                {c.desc}
              </span>
            </button>
          ))}
        </div>

        <button
          onClick={onClose}
          className="mt-5 w-full text-center text-sm text-ink-faint hover:text-ink"
        >
          Để sau
        </button>
      </div>
    </div>
  );
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
