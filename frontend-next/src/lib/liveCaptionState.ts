export interface CaptionLine {
  id: number;
  text: string;
  isFinal: boolean;
  /** ms since recording start (same timeline as server transcript_delta). */
  startMs: number;
  endMs: number;
}

export interface LiveCaptionState {
  lines: CaptionLine[];
  interim: CaptionLine | null;
  nextId: number;
}

export interface CaptionResultEvent {
  /** Newly finalized phrases, in order. */
  finals: string[];
  /** Current (not yet final) text; replaces the previous interim. */
  interim: string;
  atMs: number;
}

export const EMPTY_CAPTION_STATE: LiveCaptionState = {
  lines: [],
  interim: null,
  nextId: 0,
};

export function reduceLiveCaption(
  state: LiveCaptionState,
  event: CaptionResultEvent,
): LiveCaptionState {
  let { nextId } = state;
  // A phrase starts when its first interim word appeared.
  let openStartMs = state.interim?.startMs ?? event.atMs;
  const lines = [...state.lines];

  for (const raw of event.finals) {
    const text = raw.trim();
    if (!text) continue;
    lines.push({ id: nextId++, text, isFinal: true, startMs: openStartMs, endMs: event.atMs });
    openStartMs = event.atMs;
  }

  const interimText = event.interim.trim();
  const interim = interimText
    ? {
        id: state.interim && event.finals.length === 0 ? state.interim.id : nextId++,
        text: interimText,
        isFinal: false,
        startMs: openStartMs,
        endMs: event.atMs,
      }
    : null;

  return { lines, interim, nextId };
}

/** Keep the dangling interim as a line (session ended / recording stopped). */
export function settleInterim(state: LiveCaptionState): LiveCaptionState {
  if (!state.interim) return state;
  return reduceLiveCaption(state, {
    finals: [state.interim.text],
    interim: "",
    atMs: state.interim.endMs,
  });
}

/** Caption lines not yet covered by confirmed server transcript. */
export function unconfirmedCaptions(
  lines: CaptionLine[],
  confirmedEndMs: number,
): CaptionLine[] {
  return lines.filter((line) => (line.startMs + line.endMs) / 2 >= confirmedEndMs);
}
