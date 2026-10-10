import Link from "next/link";
import { getServerSession } from "next-auth";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import { LibraryGrid } from "@/components/LibraryGrid";
import { PageHeader } from "@/components/ui";
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
      <div className="mx-auto w-full max-w-6xl px-6 py-10 md:px-10">
        <PageHeader eyebrow="Thư viện · MeetingMind" title="Tài liệu của bạn" />
        <div className="mt-8 flex flex-col items-center justify-center rounded-2xl border border-line bg-surface py-20 text-center">
          <h3 className="font-display text-xl font-medium text-ink">Bạn đang sử dụng ẩn danh</h3>
          <p className="mt-2 max-w-md text-sm text-ink-soft">
            Ở chế độ khách, tài liệu không được lưu trữ. Vui lòng đăng nhập để quản lý thư viện của bạn.
          </p>
          <div className="mt-6 flex gap-3">
            <Link
              href="/upload"
              className="rounded-xl bg-ink px-5 py-2.5 text-sm font-medium text-white transition hover:bg-ink/90"
            >
              Tiếp tục ẩn danh
            </Link>
            <Link
              href="/login"
              className="rounded-xl border border-line bg-surface px-5 py-2.5 text-sm font-medium text-ink transition hover:bg-surface-2"
            >
              Đăng nhập
            </Link>
          </div>
        </div>
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
