// Shared types for MeetingMind Phase 2 frontend.
// These mirror the WebSocket/REST contract in meet_docs/docs/11_phase2_notebooklm_plan.md.
// The 4-person team wires these to the real FastAPI backend; the UI is built against them.

export type MediaType = "video" | "audio";

export type SourceStatus = "processing" | "done" | "failed";

export type DocMode = "live" | "summary" | "full_text";

export interface Source {
  id: string;
  title: string;
  mediaType: MediaType;
  durationMs: number;
  createdAt: string; // ISO
  status: SourceStatus;
  /** Which document modes already exist for this source. */
  docs: DocMode[];
}

export interface TranscriptSegment {
  startMs: number;
  endMs: number;
  speaker: number; // 0-based; maps to speaker palette
  text: string;
}

export interface DocSection {
  id: string;
  heading: string;
  markdown: string;
  /** True while the LLM is still writing this section (the "live" pulse). */
  writing?: boolean;
}

export type ProcessingStage =
  | "extracting_audio"
  | "transcribing"
  | "generating_doc"
  | "done";

export interface Speaker {
  id: number;
  label: string; // e.g. "Người nói 1" — editable by the user later
}
