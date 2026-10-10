"use client";

/** Optional hints that make recognition more accurate. */
export interface RecognitionHintValues {
  /** Names, terms, abbreviations — passed to Qwen3-ASR as context. */
  context: string;
  /** Number of speakers if known ("" = detect automatically). */
  speakers: string;
}

export const EMPTY_HINTS: RecognitionHintValues = { context: "", speakers: "" };

/** Parsed speaker count, or undefined when empty/invalid. */
export function hintSpeakerCount(hints: RecognitionHintValues): number | undefined {
  const value = Number.parseInt(hints.speakers, 10);
  return Number.isInteger(value) && value >= 1 && value <= 20 ? value : undefined;
}

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
  const field =
    "w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none disabled:opacity-60";
  return (
    <div className={`grid gap-3 sm:grid-cols-[1fr_9rem] ${className}`}>
      <label className="block">
        <span className="text-xs font-medium text-ink-soft">
          Từ khóa, tên riêng (tùy chọn)
        </span>
        <input
          type="text"
          value={value.context}
          maxLength={1000}
          disabled={disabled}
          onChange={(e) => onChange({ ...value, context: e.target.value })}
          placeholder="VD: anh Tuấn, chị Lan, Kubernetes, OKR"
          className={`mt-1 ${field}`}
        />
        <span className="mt-1 block text-xs text-ink-faint">
          Chỉ cần tên người, thuật ngữ, viết tắt — từ thông dụng không cần nhập.
        </span>
      </label>
      <label className="block">
        <span className="text-xs font-medium text-ink-soft">Số người nói</span>
        <input
          type="number"
          inputMode="numeric"
          min={1}
          max={20}
          value={value.speakers}
          disabled={disabled}
          onChange={(e) => onChange({ ...value, speakers: e.target.value })}
          placeholder="Tự nhận"
          className={`mt-1 ${field}`}
        />
      </label>
    </div>
  );
}
