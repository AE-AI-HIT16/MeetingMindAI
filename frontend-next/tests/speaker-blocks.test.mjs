import assert from "node:assert/strict";
import test from "node:test";

import { groupSpeakerBlocks } from "../src/lib/speakerBlocks.ts";

const seg = (speaker, startMs, text, overlapped = false) => ({ speaker, startMs, text, overlapped });
const words = (n, end = "") => Array.from({ length: n }, () => "từ").join(" ") + end;

test("consecutive segments of one speaker form one block", () => {
  const blocks = groupSpeakerBlocks([
    seg(0, 0, "Xin chào các bạn,"),
    seg(0, 4000, "hôm nay mình họp."),
    seg(1, 9000, "Dạ vâng."),
    seg(0, 10000, "Bắt đầu nhé."),
  ]);
  assert.deepEqual(
    blocks.map((b) => [b.speaker, b.startMs, b.firstIndex, b.segments.map((s) => s.text)]),
    [
      [0, 0, 0, ["Xin chào các bạn,", "hôm nay mình họp."]],
      [1, 9000, 2, ["Dạ vâng."]],
      [0, 10000, 3, ["Bắt đầu nhé."]],
    ],
  );
});

test("long monologue splits at a sentence end after the soft limit", () => {
  const blocks = groupSpeakerBlocks([
    seg(0, 0, words(40, ",")),
    seg(0, 10000, words(30, ".")), // 70 words, ends a sentence
    seg(0, 20000, words(10, ".")),
  ]);
  assert.deepEqual(blocks.map((b) => b.startMs), [0, 20000]);
});

test("run-on speech is cut at the hard limit even without punctuation", () => {
  const blocks = groupSpeakerBlocks([
    seg(0, 0, words(70)),
    seg(0, 10000, words(60)), // 130 words, no sentence end
    seg(0, 20000, words(10)),
  ]);
  assert.deepEqual(blocks.map((b) => b.startMs), [0, 20000]);
});

test("crosstalk segments get a short block of their own", () => {
  const blocks = groupSpeakerBlocks([
    seg(0, 0, "Mình bắt đầu nhé,"),
    seg(0, 3000, "phần này quan trọng", true),
    seg(0, 6000, "nên các bạn chú ý."),
  ]);
  assert.deepEqual(blocks.map((b) => b.segments.length), [1, 1, 1]);
});

test("empty transcript gives no blocks", () => {
  assert.deepEqual(groupSpeakerBlocks([]), []);
});
