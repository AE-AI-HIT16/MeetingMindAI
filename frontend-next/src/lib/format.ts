// Small formatting helpers shared across the UI.

export function formatDuration(ms: number | null): string {
  if (ms === null) return "--:--";
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** Timestamp for transcript lines: mm:ss (no leading hour unless needed). */
export function formatStamp(ms: number): string {
  return formatDuration(ms);
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

const SPEAKER_VARS = [
  "var(--color-spk-1)",
  "var(--color-spk-2)",
  "var(--color-spk-3)",
  "var(--color-spk-4)",
  "var(--color-spk-5)",
];

export function speakerColor(speaker: number): string {
  return SPEAKER_VARS[speaker % SPEAKER_VARS.length];
}

export function speakerLabel(speaker: number): string {
  return `Người nói ${speaker + 1}`;
}
