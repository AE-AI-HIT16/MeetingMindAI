"use client";

import { useEffect } from "react";
import { ErrorPage } from "@/components/ErrorPage";

// Any unexpected error while rendering a page: friendly fallback instead of
// the framework's default crash screen. Details stay in the console/logs.
export default function Error({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  useEffect(() => {
    console.error("[app error]", error);
  }, [error]);

  return (
    <ErrorPage
      title="Trang chưa tải được"
      message="Có thể máy chủ đang bận hoặc kết nối mạng bị gián đoạn. Vui lòng thử lại."
      action={
        <button
          type="button"
          onClick={() => unstable_retry()}
          className="rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-ink"
        >
          Thử lại
        </button>
      }
    />
  );
}
