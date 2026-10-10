"use client";

import { friendlyError } from "@/lib/friendlyError";
import { WarningCircle } from "@phosphor-icons/react";

/**
 * Friendly, non-alarming error message (icon + title + explanation + optional
 * "Thử lại"). Accepts any thrown value or a ready-made message string.
 */
export function ErrorNotice({
  error,
  title,
  onRetry,
  onDismiss,
  compact = false,
  className = "",
}: {
  error: unknown;
  /** Context-specific heading, e.g. "Không thể tải tài liệu". */
  title?: string;
  onRetry?: () => void;
  onDismiss?: () => void;
  /** Smaller variant for tight spots (inline editors, dialogs). */
  compact?: boolean;
  className?: string;
}) {
  if (error === null || error === undefined || error === "") return null;
  const info = friendlyError(error, title);

  return (
    <div
      role="alert"
      className={`flex items-start gap-3 rounded-xl border border-signal/25 bg-signal-wash text-left ${
        compact ? "px-3 py-2" : "px-4 py-3"
      } ${className}`}
    >
      <span
        aria-hidden="true"
        className={`mt-0.5 flex shrink-0 items-center justify-center rounded-full bg-signal/15 text-warn ${
          compact ? "h-5 w-5" : "h-7 w-7"
        }`}
      >
        <WarningCircle size={compact ? 13 : 16} weight="bold" />
      </span>
      <div className="min-w-0 flex-1">
        <p className={`font-medium text-ink ${compact ? "text-xs" : "text-sm"}`}>{info.title}</p>
        <p className={`mt-0.5 text-ink-soft ${compact ? "text-[11px]" : "text-sm"}`}>{info.message}</p>
        {(onRetry && info.retryable) || onDismiss ? (
          <div className="mt-2 flex gap-2">
            {onRetry && info.retryable && (
              <button
                type="button"
                onClick={onRetry}
                className="rounded-lg bg-ink px-3 py-1 text-xs font-medium text-paper transition hover:bg-ink/85"
              >
                Thử lại
              </button>
            )}
            {onDismiss && (
              <button
                type="button"
                onClick={onDismiss}
                className="rounded-lg border border-line px-3 py-1 text-xs font-medium text-ink-soft transition hover:bg-surface"
              >
                Đóng
              </button>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}
