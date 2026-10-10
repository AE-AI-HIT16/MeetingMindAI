/**
 * Consecutive transcript segments of one speaker shown as one readable block.
 * Segments are ASR chunks cut at pauses (often mid-sentence); they stay
 * separate inside the block so each sentence keeps its timestamp, playback
 * highlight, "nói chồng" flag and edit button.
 */
export interface SpeakerBlock<T> {
  speaker: number | null;
  startMs: number;
  /** Index of the block's first segment in the input list. */
  firstIndex: number;
  segments: T[];
}

export function groupSpeakerBlocks<
  T extends { speaker: number | null; startMs: number },
>(segments: T[]): SpeakerBlock<T>[] {
  const blocks: SpeakerBlock<T>[] = [];
  segments.forEach((segment, index) => {
    const last = blocks[blocks.length - 1];
    if (last && last.speaker === segment.speaker) {
      last.segments.push(segment);
    } else {
      blocks.push({
        speaker: segment.speaker,
        startMs: segment.startMs,
        firstIndex: index,
        segments: [segment],
      });
    }
  });
  return blocks;
}
