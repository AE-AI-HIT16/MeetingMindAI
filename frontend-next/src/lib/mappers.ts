import type {
  ApiTranscriptSegment,
  TranscriptSegment,
} from "./types";

/** Convert the backend snake_case segment into the frontend UI shape. */
export function mapTranscriptSegment(
  segment: ApiTranscriptSegment,
): TranscriptSegment {
  return {
    id: segment.id,
    startMs: segment.start_ms,
    endMs: segment.end_ms,
    speaker: segment.speaker,
    text: segment.text,
  };
}
