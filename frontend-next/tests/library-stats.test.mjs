import assert from "node:assert/strict";
import test from "node:test";

import { computeLibraryStats, formatTotalDuration } from "../src/lib/libraryStats.ts";

const src = (mediaType, durationMs, status, docs) => ({ mediaType, durationMs, status, docs });

test("library stats count media, documents and states", () => {
  const stats = computeLibraryStats([
    src("audio", 60_000, "done", ["live", "summary"]),
    src("video", 120_000, "done", ["full_text", "summary"]),
    src("audio", null, "processing", ["live"]),
    src("audio", 30_000, "failed", []),
  ]);
  assert.deepEqual(stats, {
    total: 4, audio: 3, video: 1, totalDurationMs: 210_000,
    withDocument: 2, summaries: 2, fullTexts: 1, processing: 1, failed: 1,
  });
});

test("total duration reads naturally", () => {
  assert.equal(formatTotalDuration(45_000), "45 giây");
  assert.equal(formatTotalDuration(12 * 60_000), "12 phút");
  assert.equal(formatTotalDuration(3 * 3600_000 + 5 * 60_000), "3 giờ 05 phút");
});
