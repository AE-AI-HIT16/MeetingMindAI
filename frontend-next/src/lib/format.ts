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

export interface SpeakerStyle {
  color: string;
  bg: string;
  border: string;
  gradient: string;
}

const SPEAKER_PALETTES: SpeakerStyle[] = [
  {
    // Speaker 0: Ultramarine / Indigo
    color: "#2c3ee0",
    bg: "#eef0fd",
    border: "rgba(44, 62, 224, 0.25)",
    gradient: "linear-gradient(135deg, #2c3ee0, #4f46e5)",
  },
  {
    // Speaker 1: Teal / Emerald
    color: "#0e9e8e",
    bg: "#e6f7f5",
    border: "rgba(14, 158, 142, 0.25)",
    gradient: "linear-gradient(135deg, #0e9e8e, #10b981)",
  },
  {
    // Speaker 2: Amber / Coral
    color: "#b3591d",
    bg: "#fef3eb",
    border: "rgba(179, 89, 29, 0.25)",
    gradient: "linear-gradient(135deg, #b3591d, #f59e0b)",
  },
  {
    // Speaker 3: Violet / Purple
    color: "#7c5cfc",
    bg: "#f3f0ff",
    border: "rgba(124, 92, 252, 0.25)",
    gradient: "linear-gradient(135deg, #7c5cfc, #8b5cf6)",
  },
  {
    // Speaker 4: Rose / Pink
    color: "#c02c74",
    bg: "#fce8f3",
    border: "rgba(192, 44, 116, 0.25)",
    gradient: "linear-gradient(135deg, #c02c74, #ec4899)",
  },
  {
    // Speaker 5: Cyan / Sky
    color: "#0284c7",
    bg: "#e0f2fe",
    border: "rgba(2, 132, 199, 0.25)",
    gradient: "linear-gradient(135deg, #0284c7, #38bdf8)",
  },
];

export function getSpeakerStyle(speaker: number | null): SpeakerStyle {
  if (speaker === null || speaker === undefined) {
    return {
      color: "#6b7280",
      bg: "#f3f4f6",
      border: "rgba(107, 114, 128, 0.2)",
      gradient: "linear-gradient(135deg, #6b7280, #9ca3af)",
    };
  }
  const index = Math.abs(speaker) % SPEAKER_PALETTES.length;
  return SPEAKER_PALETTES[index];
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
