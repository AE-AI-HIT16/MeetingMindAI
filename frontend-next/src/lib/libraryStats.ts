import type { Source } from "./types";

export interface LibraryStats {
  total: number;
  audio: number;
  video: number;
  totalDurationMs: number;
  /** Sources with a finished summary or full-text document. */
  withDocument: number;
  summaries: number;
  fullTexts: number;
  processing: number;
  failed: number;
}

export function computeLibraryStats(sources: Source[]): LibraryStats {
  const stats: LibraryStats = {
    total: sources.length,
    audio: 0,
    video: 0,
    totalDurationMs: 0,
    withDocument: 0,
    summaries: 0,
    fullTexts: 0,
    processing: 0,
    failed: 0,
  };
  for (const source of sources) {
    if (source.mediaType === "video") stats.video += 1;
    else stats.audio += 1;
    stats.totalDurationMs += source.durationMs ?? 0;
    const hasSummary = source.docs.includes("summary");
    const hasFullText = source.docs.includes("full_text");
    if (hasSummary) stats.summaries += 1;
    if (hasFullText) stats.fullTexts += 1;
    if (hasSummary || hasFullText) stats.withDocument += 1;
    if (source.status === "processing") stats.processing += 1;
    if (source.status === "failed") stats.failed += 1;
  }
  return stats;
}

/** "3 giờ 05 phút", "12 phút", "45 giây". */
export function formatTotalDuration(ms: number): string {
  const seconds = Math.round(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours} giờ ${String(minutes).padStart(2, "0")} phút`;
  if (minutes > 0) return `${minutes} phút`;
  return `${seconds} giây`;
}
