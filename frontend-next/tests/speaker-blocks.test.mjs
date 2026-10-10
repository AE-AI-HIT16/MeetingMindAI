import assert from "node:assert/strict";
import test from "node:test";

import { groupSpeakerBlocks } from "../src/lib/speakerBlocks.ts";

const seg = (speaker, startMs, text) => ({ speaker, startMs, text });

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

test("empty transcript gives no blocks", () => {
  assert.deepEqual(groupSpeakerBlocks([]), []);
});
