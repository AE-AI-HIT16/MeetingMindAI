import Link from "next/link";
import { MOCK_SOURCES } from "@/lib/mock";
import { SourceCard } from "@/components/SourceCard";
import { PageHeader } from "@/components/ui";

export default function LibraryPage() {
  const sources = MOCK_SOURCES;
  const processing = sources.filter((s) => s.status === "processing").length;

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-10 md:px-10">
      <PageHeader eyebrow="Thư viện · MeetingMind" title="Tài liệu của bạn">
        <Link
          href="/upload"
          className="flex items-center gap-2 rounded-xl bg-ink px-4 py-2.5 text-sm font-medium text-white transition hover:bg-ink/90"
        >
          Tải lên media
        </Link>
      </PageHeader>

      <p className="mt-3 max-w-xl text-sm text-ink-soft">
        Tải lên video hoặc audio, xem tài liệu hình thành theo thời gian thực,
        rồi chọn tóm tắt hay giữ toàn văn.
        {processing > 0 && (
          <>
            {" "}
            Hiện có{" "}
            <span className="font-medium text-signal-ink">
              {processing} tài liệu đang xử lý
            </span>
            .
          </>
        )}
      </p>

      <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <Link
          href="/upload"
          className="group flex min-h-[260px] flex-col items-center justify-center rounded-[var(--radius-card)] border-2 border-dashed border-line bg-surface/50 p-5 text-center transition hover:border-brand hover:bg-brand-wash/40"
        >
          <span className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-wash text-brand transition group-hover:scale-110">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </span>
          <span className="mt-3 font-display text-lg font-medium text-ink">
            Tài liệu mới
          </span>
          <span className="mt-1 text-xs text-ink-faint">
            Kéo thả hoặc chọn tệp media
          </span>
        </Link>

        {sources.map((s) => (
          <SourceCard key={s.id} source={s} />
        ))}
      </div>
    </div>
  );
}
