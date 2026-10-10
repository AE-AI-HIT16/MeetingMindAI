import Link from "next/link";
import { getServerSession } from "next-auth";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import { LibraryGrid } from "@/components/LibraryGrid";
import { StartPanel } from "@/components/StartPanel";
import { PageTitle } from "@/components/ui";
import { ArrowRight, Brain, FileArrowDown, LockSimple, UsersThree } from "@phosphor-icons/react/dist/ssr";
import { APIError, listSources } from "@/lib/api";
import { ErrorNotice } from "@/components/ErrorNotice";
import { friendlyError } from "@/lib/friendlyError";

// Luôn render động — không bao giờ tạo tĩnh lúc build.
// Vì trang này cần session (auth) và kết nối backend (RunPod).
export const dynamic = "force-dynamic";

export default async function LibraryPage() {
  const session = await getServerSession(authOptions);
  const token = session?.accessToken;

  // Nếu là khách, không tải library
  if (!session) {
    return (
      <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 md:px-10 md:py-14">
        <div className="animate-enter">
          <PageTitle size="lg" title="Cứ họp đi." lede="Ghi chép để MeetingMind lo." />
        </div>

        <div className="animate-enter mt-10 [animation-delay:80ms]">
          <StartPanel large />
        </div>

        <div className="animate-enter mt-4 flex flex-col gap-4 rounded-[var(--radius-bezel)] bg-surface/60 p-5 ring-1 ring-line/80 backdrop-blur sm:flex-row sm:items-center [animation-delay:160ms]">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-surface-3 text-ink-soft ring-1 ring-line">
            <LockSimple size={18} />
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-ink">Bạn đang dùng chế độ khách</p>
            <p className="mt-0.5 text-sm text-ink-soft">
              Tài liệu không được lưu lại. Đăng nhập để có thư viện riêng.
            </p>
          </div>
          <Link href="/login" className="btn btn-primary shrink-0 self-start sm:self-auto">
            Đăng nhập
            <span className="btn-nest">
              <ArrowRight size={14} weight="bold" />
            </span>
          </Link>
        </div>

        <ol className="animate-enter mt-14 grid gap-8 border-t border-line pt-8 md:grid-cols-3 [animation-delay:240ms]">
          {STEPS.map(({ icon: StepIcon, title, desc }) => (
            <li key={title} className="flex gap-4">
              <StepIcon size={22} className="mt-0.5 shrink-0 text-brand" />
              <div>
                <p className="font-medium text-ink">{title}</p>
                <p className="mt-1 text-sm leading-relaxed text-ink-faint">{desc}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  const { sources, loadError } = await listSources(token)
    .then((items) => ({ sources: items, loadError: null }))
    .catch((error: unknown) => {
      console.error("[LibraryPage Server Error] listSources failed:", error);
      return {
        sources: [],
        // Friendly text only; technical detail stays in the server log.
        loadError: friendlyError(error, "Chưa tải được thư viện"),
      };
    });
  const processing = sources.filter((s) => s.status === "processing").length;

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 md:px-10 md:py-12">
      <div className="animate-enter flex flex-wrap items-end justify-between gap-4">
        <div>
          <PageTitle title="Thư viện" lede="Mọi cuộc họp, gọn một chỗ." />
          {processing > 0 && (
            <p className="mt-3 inline-flex items-center gap-2 text-sm font-medium text-signal-ink">
              <span className="recording-dot h-1.5 w-1.5 rounded-full bg-signal" />
              {processing} tài liệu đang xử lý
            </p>
          )}
        </div>
      </div>

      <div className="animate-enter mt-8 [animation-delay:60ms]">
        <StartPanel />
      </div>

      {loadError && (
        <ErrorNotice
          error={loadError.message}
          title={loadError.title}
          className="mt-6"
        />
      )}

      <LibraryGrid sources={sources} />
    </div>
  );
}

const STEPS = [
  {
    icon: UsersThree,
    title: "Tách người nói",
    desc: "Nhận ra ai đang nói, đánh dấu những đoạn nói chồng.",
  },
  {
    icon: Brain,
    title: "Dựng tài liệu",
    desc: "Nội dung được sắp thành mục ngay khi lời thoại về tới.",
  },
  {
    icon: FileArrowDown,
    title: "Xuất bản",
    desc: "Chọn tóm tắt hoặc toàn văn, rồi tải PDF, DOCX hay Markdown.",
  },
];
