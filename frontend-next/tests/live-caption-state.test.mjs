import assert from "node:assert/strict";
import test from "node:test";

import {
  EMPTY_CAPTION_STATE,
  reduceLiveCaption,
  settleInterim,
  unconfirmedCaptions,
} from "../src/lib/liveCaptionState.ts";

function result(finals, interim, atMs) {
  return { finals, interim, atMs };
}

test("interim replaces previous interim instead of appending", () => {
  let state = reduceLiveCaption(EMPTY_CAPTION_STATE, result([], "xin", 100));
  state = reduceLiveCaption(state, result([], "xin chào", 300));

  assert.equal(state.lines.length, 0);
  assert.equal(state.interim.text, "xin chào");
  assert.equal(state.interim.startMs, 100);
  assert.equal(state.interim.endMs, 300);
});

test("final moves phrase into lines and clears interim", () => {
  let state = reduceLiveCaption(EMPTY_CAPTION_STATE, result([], "xin chào", 100));
  const interimId = state.interim.id;
  state = reduceLiveCaption(state, result(["xin chào mọi người"], "", 900));

  assert.equal(state.interim, null);
  assert.deepEqual(
    state.lines.map(({ text, startMs, endMs }) => ({ text, startMs, endMs })),
    [{ text: "xin chào mọi người", startMs: 100, endMs: 900 }],
  );
  assert.notEqual(state.lines[0].id, undefined);
  assert.ok(state.nextId > interimId);
});

test("multiple finals and a new interim in one event", () => {
  const state = reduceLiveCaption(
    EMPTY_CAPTION_STATE,
    result(["câu một", "câu hai"], "câu ba", 2000),
  );

  assert.deepEqual(state.lines.map((l) => l.text), ["câu một", "câu hai"]);
  assert.equal(state.interim.text, "câu ba");
  const ids = [...state.lines.map((l) => l.id), state.interim.id];
  assert.equal(new Set(ids).size, ids.length);
});

test("blank finals are ignored", () => {
  const state = reduceLiveCaption(EMPTY_CAPTION_STATE, result(["  ", ""], " ", 500));
  assert.equal(state.lines.length, 0);
  assert.equal(state.interim, null);
});

test("lines survive across recognition restarts", () => {
  let state = reduceLiveCaption(EMPTY_CAPTION_STATE, result(["trước restart"], "", 1000));
  // After a session restart the hook drops the interim and keeps lines.
  state = { ...state, interim: null };
  state = reduceLiveCaption(state, result(["sau restart"], "", 5000));

  assert.deepEqual(state.lines.map((l) => l.text), ["trước restart", "sau restart"]);
});

test("unconfirmedCaptions hides lines covered by server transcript", () => {
  const lines = [
    { id: 0, text: "a", isFinal: true, startMs: 0, endMs: 4000 },
    { id: 1, text: "b", isFinal: true, startMs: 13000, endMs: 17000 },
    { id: 2, text: "c", isFinal: true, startMs: 17000, endMs: 20000 },
  ];

  assert.deepEqual(unconfirmedCaptions(lines, 0).map((l) => l.text), ["a", "b", "c"]);
  assert.deepEqual(unconfirmedCaptions(lines, 15000).map((l) => l.text), ["b", "c"]);
  assert.deepEqual(unconfirmedCaptions(lines, 30000), []);
});

test("settleInterim keeps the visible interim as a line", () => {
  let state = reduceLiveCaption(EMPTY_CAPTION_STATE, result(["đã xong"], "đang nói dở", 1200));
  state = settleInterim(state);

  assert.equal(state.interim, null);
  assert.deepEqual(state.lines.map((l) => l.text), ["đã xong", "đang nói dở"]);
  assert.equal(settleInterim(state), state);
});
