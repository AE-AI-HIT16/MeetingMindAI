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
}

// Readable on white: deep text on a light tint of the same hue.
const SPEAKER_PALETTES: SpeakerStyle[] = [
  {
    // Speaker 0: Ultramarine
    color: "#3b4fd8",
    bg: "rgb(59 79 216 / 0.08)",
    border: "rgb(59 79 216 / 0.22)",
  },
  {
    // Speaker 1: Teal
    color: "#0a7f72",
    bg: "rgb(10 127 114 / 0.08)",
    border: "rgb(10 127 114 / 0.22)",
  },
  {
    // Speaker 2: Amber
    color: "#a2551b",
    bg: "rgb(162 85 27 / 0.08)",
    border: "rgb(162 85 27 / 0.22)",
  },
  {
    // Speaker 3: Violet
    color: "#6d45e0",
    bg: "rgb(109 69 224 / 0.08)",
    border: "rgb(109 69 224 / 0.22)",
  },
  {
    // Speaker 4: Rose
    color: "#b42a6a",
    bg: "rgb(180 42 106 / 0.08)",
    border: "rgb(180 42 106 / 0.22)",
  },
  {
    // Speaker 5: Sky
    color: "#0369a1",
    bg: "rgb(3 105 161 / 0.08)",
    border: "rgb(3 105 161 / 0.22)",
  },
];

export function getSpeakerStyle(speaker: number | null): SpeakerStyle {
  if (speaker === null || speaker === undefined) {
    return {
      color: "#4f545e",
      bg: "rgb(79 84 94 / 0.07)",
      border: "rgb(79 84 94 / 0.18)",
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
