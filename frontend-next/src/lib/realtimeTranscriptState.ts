export type RealtimeTranscriptType =
  | "transcript_delta"
  | "transcript_partial";

export interface SentenceInfo {
  text: string;
  start: number;
  end: number;
  speaker: string | number | null;
}

export interface TranscriptDelta {
  text: string;
  sentenceInfo: SentenceInfo[];
  receivedAt: number;
  startMs: number;
  endMs: number;
  speaker: number | null;
  type: RealtimeTranscriptType;
}

export interface RealtimeTranscriptEvent {
  type: RealtimeTranscriptType;
  receivedAt: number;
  segment: {
    text: string;
    startMs: number;
    endMs: number;
    speaker: number | null;
  };
}

function toTranscriptItem(event: RealtimeTranscriptEvent): TranscriptDelta {
  const { segment } = event;
  return {
    text: segment.text,
    sentenceInfo: [
      {
        text: segment.text,
        start: segment.startMs / 1000,
        end: segment.endMs / 1000,
        speaker: segment.speaker,
      },
    ],
    receivedAt: event.receivedAt,
    startMs: segment.startMs,
    endMs: segment.endMs,
    speaker: segment.speaker,
    type: event.type,
  };
}

function confirmedDeltaBelongsToPartial(
  delta: TranscriptDelta,
  partial: TranscriptDelta,
): boolean {
  // Confirmed ASR may start a little after the VAD utterance boundary. The
  // delta still belongs to this partial when its first timestamp is inside it.
  return (
    partial.startMs <= delta.startMs && delta.startMs < partial.endMs
  );
}

function sortByTimeline(items: TranscriptDelta[]): TranscriptDelta[] {
  return items.sort(
    (left, right) =>
      left.startMs - right.startMs ||
      left.endMs - right.endMs ||
      left.type.localeCompare(right.type),
  );
}

export function reduceRealtimeTranscripts(
  state: TranscriptDelta[],
  event: RealtimeTranscriptEvent,
): TranscriptDelta[] {
  const item = toTranscriptItem(event);

  if (item.type === "transcript_partial") {
    const alreadyConfirmed = state.some(
      (current) =>
        current.type === "transcript_delta" &&
        confirmedDeltaBelongsToPartial(current, item),
    );
    if (alreadyConfirmed) {
      return state;
    }

    const currentPartial = state.find(
      (current) =>
        current.type === "transcript_partial" &&
        current.startMs === item.startMs,
    );
    if (currentPartial && currentPartial.endMs > item.endMs) {
      return state;
    }

    return sortByTimeline([
      ...state.filter(
        (current) =>
          current.type !== "transcript_partial" ||
          current.startMs !== item.startMs,
      ),
      item,
    ]);
  }

  return sortByTimeline([
    ...state.filter(
      (current) =>
        !(
          current.type === "transcript_delta" &&
          current.startMs === item.startMs
        ) &&
        !(
          current.type === "transcript_partial" &&
          confirmedDeltaBelongsToPartial(item, current)
        ),
    ),
    item,
  ]);
}
