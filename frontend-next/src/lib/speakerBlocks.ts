/**
 * Consecutive transcript segments of one speaker shown as readable blocks.
 * Segments are ASR chunks cut at pauses (often mid-sentence); they stay
 * separate inside a block so each sentence keeps its timestamp, playback
 * highlight, "nói chồng" flag and edit button.
 *
 * Blocks stay short enough to follow: a long monologue is split at the end of
 * a sentence once it reaches ``softWords`` (always by ``maxWords``), and
 * crosstalk segments get a block of their own.
 */
export interface SpeakerBlock<T> {
  speaker: number | null;
  startMs: number;
  /** Index of the block's first segment in the input list. */
  firstIndex: number;
  segments: T[];
}

export interface BlockLimits {
  softWords?: number;
  maxWords?: number;
}

const SENTENCE_END = /[.?!…]["')\]]*\s*$/;

function wordCount(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

export function groupSpeakerBlocks<
  T extends { speaker: number | null; startMs: number; text: string; overlapped?: boolean },
>(segments: T[], { softWords = 60, maxWords = 120 }: BlockLimits = {}): SpeakerBlock<T>[] {
  const blocks: SpeakerBlock<T>[] = [];
  let words = 0;
  segments.forEach((segment, index) => {
    const last = blocks[blocks.length - 1];
    const previous = last?.segments[last.segments.length - 1];
    const continues =
      last !== undefined &&
      last.speaker === segment.speaker &&
      !segment.overlapped &&
      !previous?.overlapped &&
      words < maxWords &&
      !(words >= softWords && SENTENCE_END.test(previous?.text ?? ""));
    if (continues) {
      last.segments.push(segment);
      words += wordCount(segment.text);
    } else {
      blocks.push({
        speaker: segment.speaker,
        startMs: segment.startMs,
        firstIndex: index,
        segments: [segment],
      });
      words = wordCount(segment.text);
    }
  });
  return blocks;
}
