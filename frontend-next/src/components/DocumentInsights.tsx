"use client";

import { useMemo } from "react";
import type { TranscriptSegment } from "@/lib/types";
import { formatDuration, getSpeakerStyle } from "@/lib/format";
import { SpeakerChip } from "@/components/ui";

type Focus = { speaker: number | null } | null;

/**
 * Side panel for the result page: meeting facts and who spoke how much,
 * computed from the transcript segments. On full-text documents a speaker
 * can be picked to bring only their turns forward.
 */
export function DocumentInsights({
  segments,
  durationMs,
  focus,
  onFocus,
}: {
  segments: TranscriptSegment[];
  durationMs: number | null;
  focus: Focus;
  /** Present only when the document has turns to highlight. */
  onFocus?: (focus: Focus) => void;
}) {
  const stats = useMemo(() => {
    const bySpeaker = new Map<string, { speaker: number | null; ms: number; turns: number }>();
    let words = 0;
    let previous: string | null = null;
    for (const seg of segments) {
      const key = String(seg.speaker);
      const entry = bySpeaker.get(key) ?? { speaker: seg.speaker, ms: 0, turns: 0 };
      entry.ms += Math.max(0, seg.endMs - seg.startMs);
      // A turn starts whenever the speaker changes.
      if (key !== previous) entry.turns += 1;
      previous = key;
      bySpeaker.set(key, entry);
      words += seg.text.split(/\s+/).filter(Boolean).length;
    }
    const speakers = [...bySpeaker.values()].sort((a, b) => b.ms - a.ms);
    const talkMs = speakers.reduce((sum, s) => sum + s.ms, 0);
    return { speakers, talkMs, words };
  }, [segments]);

  if (segments.length === 0) return null;

  const known = stats.speakers.filter((s) => s.speaker !== null).length;
  const facts = [
    { label: "Thời lượng", value: formatDuration(durationMs) },
    { label: "Người nói", value: String(known) },
    { label: "Số từ", value: stats.words.toLocaleString("vi-VN") },
    { label: "Đọc trong", value: `${Math.max(1, Math.round(stats.words / 220))} phút` },
  ];

  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl bg-line-soft ring-1 ring-line-soft">
        {facts.map((f) => (
          <div key={f.label} className="bg-surface px-3.5 py-3">
            <dt className="text-[11px] font-medium text-ink-faint">{f.label}</dt>
            <dd className="mt-0.5 font-mono text-base font-medium tabular-nums text-ink">{f.value}</dd>
          </div>
        ))}
      </dl>

      <div>
        <div className="flex items-baseline justify-between">
          <p className="text-xs font-semibold text-ink-faint">Người tham gia</p>
          {onFocus && focus && (
            <button
              type="button"
              onClick={() => onFocus(null)}
              className="text-[11px] font-medium text-brand-ink hover:underline"
            >
              Hiện tất cả
            </button>
          )}
        </div>
        <ul className="mt-3 space-y-1">
          {stats.speakers.map((s) => {
            const share = stats.talkMs > 0 ? s.ms / stats.talkMs : 0;
            const active = focus !== null && focus.speaker === s.speaker;
            const row = (
              <>
                <div className="flex items-center justify-between gap-2">
                  <SpeakerChip speaker={s.speaker} />
                  <span className="font-mono text-[11px] tabular-nums text-ink-faint">
                    {Math.round(share * 100)}%
                  </span>
                </div>
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-line-soft">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${share * 100}%`, backgroundColor: getSpeakerStyle(s.speaker).color }}
                  />
                </div>
                <p className="mt-1.5 text-[11px] text-ink-faint">
                  {formatDuration(s.ms)} · {s.turns} lượt
                </p>
              </>
            );
            return (
              <li key={String(s.speaker)}>
                {onFocus ? (
                  <button
                    type="button"
                    aria-pressed={active}
                    onClick={() => onFocus(active ? null : { speaker: s.speaker })}
                    className={`w-full rounded-xl px-3 py-2.5 text-left transition ${
                      active ? "bg-surface shadow-[var(--shadow-card)] ring-1 ring-line" : "hover:bg-surface/70"
                    }`}
                  >
                    {row}
                  </button>
                ) : (
                  <div className="px-3 py-2.5">{row}</div>
                )}
              </li>
            );
          })}
        </ul>
        {onFocus && !focus && (
          <p className="mt-2 px-3 text-[11px] leading-relaxed text-ink-faint">
            Bấm vào một người để chỉ làm nổi lời của người đó.
          </p>
        )}
      </div>
    </div>
  );
}
