import type { SourceStatus } from "@/lib/types";
import { getSpeakerStyle, speakerLabel } from "@/lib/format";

/* ---------- Page header ---------- */
export function PageHeader({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="font-display mt-2 text-3xl font-semibold text-ink">
          {title}
        </h1>
      </div>
      {children}
    </div>
  );
}

/* ---------- Status badge ---------- */
const STATUS: Record<
  SourceStatus,
  { label: string; dot: string; text: string; bg: string }
> = {
  processing: {
    label: "Đang xử lý",
    dot: "bg-signal",
    text: "text-signal-ink",
    bg: "bg-signal-wash",
  },
  done: {
    label: "Hoàn tất",
    dot: "bg-ok",
    text: "text-ok",
    bg: "bg-[color-mix(in_srgb,var(--color-ok)_12%,#fff)]",
  },
  failed: {
    label: "Lỗi",
    dot: "bg-danger",
    text: "text-danger",
    bg: "bg-[color-mix(in_srgb,var(--color-danger)_10%,#fff)]",
  },
};

export function StatusBadge({ status }: { status: SourceStatus }) {
  const s = STATUS[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${s.bg} ${s.text}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${s.dot} ${
          status === "processing" ? "animate-pulse" : ""
        }`}
      />
      {s.label}
    </span>
  );
}

/* ---------- Speaker chip ---------- */
export function SpeakerChip({ speaker }: { speaker: number | null }) {
  if (speaker === null || speaker === undefined) {
    return (
      <span className="font-mono text-xs text-ink-faint">
        Chưa xác định
      </span>
    );
  }

  const style = getSpeakerStyle(speaker);
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 font-mono text-xs font-semibold"
      style={{
        backgroundColor: style.bg,
        color: style.color,
        border: `1px solid ${style.border}`,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: style.color }}
      />
      {speakerLabel(speaker)}
    </span>
  );
}

/* ---------- Waveform (the signature motif) ---------- */
export function Waveform({
  live = false,
  bars = 28,
  className = "",
}: {
  live?: boolean;
  bars?: number;
  className?: string;
}) {
  // Deterministic pseudo-heights so SSR and client agree (no Math.random).
  const heights = Array.from({ length: bars }, (_, i) => {
    const v = Math.abs(Math.sin(i * 1.7) * 0.6 + Math.cos(i * 0.9) * 0.4);
    return 0.22 + (v % 1) * 0.78;
  });
  return (
    <div
      className={`flex items-center gap-[3px] ${className}`}
      aria-hidden
    >
      {heights.map((h, i) => (
        <span
          key={i}
          className="w-[3px] flex-1 rounded-full"
          style={{
            height: `${h * 100}%`,
            backgroundColor: live ? "var(--color-signal)" : "var(--color-line)",
            transformOrigin: "center",
            animation: live
              ? `wave ${0.9 + (i % 5) * 0.12}s ease-in-out ${
                  (i % 7) * 0.08
                }s infinite`
              : undefined,
          }}
        />
      ))}
    </div>
  );
}
