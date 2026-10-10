"use client";

import Link from "next/link";
import "./globals.css";

// Errors in the root layout itself: must render its own <html>/<body>.
export default function GlobalError({
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  return (
    <html lang="vi">
      <body className="min-h-full bg-paper">
        <title>MeetingMind - Đã có lỗi</title>
        <div className="mx-auto flex min-h-dvh max-w-md flex-col items-center justify-center px-6 text-center">
          <h1 className="text-xl font-semibold text-ink">Ứng dụng gặp sự cố</h1>
          <p className="mt-2 text-sm text-ink-soft">
            Vui lòng tải lại trang. Nếu lỗi vẫn tiếp diễn, hãy thử lại sau ít phút.
          </p>
          <div className="mt-6 flex gap-3">
            <button
              type="button"
              onClick={() => unstable_retry()}
              className="btn btn-primary"
            >
              Thử lại
            </button>
            <Link href="/" className="btn btn-secondary">
              Về thư viện
            </Link>
          </div>
        </div>
      </body>
    </html>
  );
}
