"use client";

/** Optional hints that make recognition more accurate. */
export interface RecognitionHintValues {
  /** Names, terms, abbreviations — passed to Qwen3-ASR as context. */
  context: string;
}

export const EMPTY_HINTS: RecognitionHintValues = { context: "" };

export function RecognitionHints({
  value,
  onChange,
  disabled = false,
  className = "",
}: {
  value: RecognitionHintValues;
  onChange: (value: RecognitionHintValues) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <label className={`block ${className}`}>
      <span className="text-xs font-medium text-ink-soft">
        Từ khóa, tên riêng (tùy chọn)
      </span>
      <input
        type="text"
        value={value.context}
        maxLength={1000}
        disabled={disabled}
        onChange={(e) => onChange({ context: e.target.value })}
        placeholder="VD: anh Tuấn, chị Lan, Kubernetes, OKR"
        className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none disabled:opacity-60"
      />
      <span className="mt-1 block text-xs text-ink-faint">
        Chỉ cần tên người, thuật ngữ, viết tắt — từ thông dụng không cần nhập.
        Số người nói được nhận tự động.
      </span>
    </label>
  );
}
