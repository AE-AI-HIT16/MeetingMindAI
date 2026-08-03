import assert from "node:assert/strict";
import test from "node:test";

import { reduceRealtimeTranscripts } from "../src/lib/realtimeTranscriptState.ts";

function event(type, startMs, endMs, text, receivedAt = endMs) {
  return {
    type,
    receivedAt,
    segment: { startMs, endMs, speaker: null, text },
  };
}

test("new partial replaces only the same utterance", () => {
  let state = [];
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 100, 700, "xin chào"),
  );
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 100, 1300, "xin chào mọi người"),
  );
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 2000, 2500, "câu thứ hai"),
  );

  assert.deepEqual(
    state.map(({ type, startMs, endMs, text }) => ({
      type,
      startMs,
      endMs,
      text,
    })),
    [
      {
        type: "transcript_partial",
        startMs: 100,
        endMs: 1300,
        text: "xin chào mọi người",
      },
      {
        type: "transcript_partial",
        startMs: 2000,
        endMs: 2500,
        text: "câu thứ hai",
      },
    ],
  );
});

test("stale partial cannot roll an utterance backward", () => {
  let state = reduceRealtimeTranscripts(
    [],
    event("transcript_partial", 100, 1300, "bản mới"),
  );
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 100, 900, "bản cũ đến trễ"),
  );

  assert.equal(state.length, 1);
  assert.equal(state[0].endMs, 1300);
  assert.equal(state[0].text, "bản mới");
});

test("confirmed delta removes only its utterance and rejects its late partial", () => {
  let state = [];
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 100, 1800, "partial câu một"),
  );
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 2200, 2800, "partial câu hai"),
  );
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_delta", 250, 1700, "final câu một"),
  );

  assert.deepEqual(
    state.map(({ type, startMs, text }) => ({ type, startMs, text })),
    [
      { type: "transcript_delta", startMs: 250, text: "final câu một" },
      {
        type: "transcript_partial",
        startMs: 2200,
        text: "partial câu hai",
      },
    ],
  );

  const afterLatePartial = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 100, 1900, "partial cũ đến trễ"),
  );
  assert.deepEqual(afterLatePartial, state);
});

test("new utterance starting at the previous confirmed end is accepted", () => {
  let state = reduceRealtimeTranscripts(
    [],
    event("transcript_delta", 0, 1000, "đã chốt"),
  );
  state = reduceRealtimeTranscripts(
    state,
    event("transcript_partial", 1000, 1500, "câu mới"),
  );

  assert.equal(state.length, 2);
  assert.equal(state[1].type, "transcript_partial");
  assert.equal(state[1].text, "câu mới");
});
