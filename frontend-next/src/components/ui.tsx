import type { SourceStatus } from "@/lib/types";
import { getSpeakerStyle, speakerLabel } from "@/lib/format";

/* ---------- Page title ---------- */
/** Short display title with a one-line lede underneath: two clear tiers
 *  instead of one long run-on headline. */
export function PageTitle({
  title,
  lede,
  size = "md",
}: {
  title: string;
  lede?: string;
  size?: "md" | "lg";
}) {
  return (
    <div>
      <h1
        className={`font-display font-semibold tracking-[-0.04em] text-ink ${
          size === "lg"
            ? "text-[44px] leading-[1] sm:text-6xl md:text-7xl"
            : "text-4xl leading-[1.05] md:text-5xl"
        }`}
      >
        {title}
      </h1>
      {lede && (
        <p className={`max-w-xl text-ink-soft ${size === "lg" ? "mt-5 text-lg" : "mt-3 text-base"}`}>
          {lede}
        </p>
      )}
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
    bg: "bg-ok-wash",
  },
  failed: {
    label: "Lỗi",
    dot: "bg-danger",
    text: "text-danger",
    bg: "bg-danger-wash",
  },
};

export function StatusBadge({ status }: { status: SourceStatus }) {
  const s = STATUS[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium ${s.bg} ${s.text}`}
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

/* ---------- Document mode badge ---------- */
/** Summary and full text get distinct hues so they read apart at a glance. */
const DOC_MODE: Record<string, { label: string; className: string }> = {
  summary: {
    label: "Tóm tắt",
    className: "bg-brand-wash text-brand-ink ring-brand/20",
  },
  full_text: {
    label: "Toàn văn",
    className: "bg-[rgb(59_79_216/0.08)] text-spk-1 ring-spk-1/20",
  },
  live: {
    label: "Đang tạo",
    className: "bg-surface-2 text-ink-soft ring-line",
  },
};

export function DocModeBadge({ mode }: { mode: string }) {
  const m = DOC_MODE[mode];
  if (!m) return null;
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold ring-1 ${m.className}`}>
      {m.label}
    </span>
  );
}

/* ---------- Speaker chip ---------- */
export function SpeakerChip({ speaker }: { speaker: number | null }) {
  if (speaker === null || speaker === undefined) {
    // Same chip shape as known speakers, dashed to read as "not identified".
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md border border-dashed border-ink-faint/50 px-2 py-0.5 font-mono text-xs font-semibold text-ink-faint">
        <span className="h-1.5 w-1.5 rounded-full border border-ink-faint/70" />
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
            // Rounded: Math.sin can differ in the last digits between the
            // server and the browser, which breaks hydration.
            height: `${(h * 100).toFixed(1)}%`,
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

/** Marks a transcript segment where two people talk at once. */
export function CrosstalkBadge() {
  return (
    <span
      title="Có người khác nói chồng lên đoạn này, chữ có thể thiếu hoặc lẫn"
      className="inline-flex items-center rounded-md border border-line px-1.5 py-0.5 text-[11px] font-medium text-ink-soft"
    >
      Nói chồng
    </span>
  );
}
