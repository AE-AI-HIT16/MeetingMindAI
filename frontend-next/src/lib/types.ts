// Shared types for MeetingMind Phase 2 frontend.
// These mirror the WebSocket/REST contract in meet_docs/docs/11_phase2_notebooklm_plan.md.
// The 4-person team wires these to the real FastAPI backend; the UI is built against them.

export type MediaType = "video" | "audio";

export type SourceStatus = "processing" | "done" | "failed";

export type DocMode = "live" | "summary" | "full_text";

/** Response returned after a Source and its processing Job are created. */
export interface CreateSourceResponse {
  sourceId: string;
  jobId: string;
  status: "queued";
}

export interface SourceDocumentRef {
  id: string;
  mode: DocMode;
}

export interface Source {
  id: string;
  title: string;
  mediaType: MediaType;
  durationMs: number | null;
  createdAt: string; // ISO
  status: SourceStatus;
  jobId: string | null;
  /** Which document modes already exist for this source. */
  docs: DocMode[];
  /** IDs needed to reopen a persisted document from the Library. */
  documents: SourceDocumentRef[];
}

/** Transcript segment exactly as returned by the backend API/WebSocket. */
export interface ApiTranscriptSegment {
  id: number | null;
  start_ms: number;
  end_ms: number;
  speaker: number | null;
  text: string;
}

/** Transcript segment normalized for React components. */
export interface TranscriptSegment {
  id: number | null;
  startMs: number;
  endMs: number;
  speaker: number | null; // 0-based; null until diarization completes
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

export interface StatusJobEvent {
  type: "status";
  stage: Exclude<ProcessingStage, "done">;
  progress: number;
}

export interface TranscriptDeltaJobEvent {
  type: "transcript_delta";
  segment: ApiTranscriptSegment;
}

export interface DocDeltaJobEvent {
  type: "doc_delta";
  section_id: string;
  markdown: string;
}

export interface SpeakerUpdateJobEvent {
  type: "speaker_update";
  updates: {
    segment_id: number;
    speaker: number | null;
  }[];
}

export interface DoneJobEvent {
  type: "done";
  duration_ms: number;
  num_segments: number;
  live_document_id: string | null;
}

export interface ErrorJobEvent {
  type: "error";
  code: string;
  message: string;
}

export type JobEvent =
  | StatusJobEvent
  | TranscriptDeltaJobEvent
  | DocDeltaJobEvent
  | SpeakerUpdateJobEvent
  | DoneJobEvent
  | ErrorJobEvent;

export interface DocumentData {
  id: string;
  sourceId: string;
  mode: DocMode;
  markdown: string;
  createdAt: string;
  updatedAt: string;
}

export type DocumentGenerationStatus =
  | "queued"
  | "processing"
  | "done"
  | "failed";

export type DocumentGenerationStage =
  | "queued"
  | "generating"
  | "saving"
  | "done"
  | "failed";

export interface DocumentGeneration {
  generationJobId: string;
  documentId: string;
  sourceId: string;
  mode: Exclude<DocMode, "live">;
  status: DocumentGenerationStatus;
  stage: DocumentGenerationStage;
  progress: number;
  error: string | null;
}

export interface DocumentGenerationStatusEvent {
  type: "document_status";
  generation_job_id: string;
  document_id: string;
  stage: Extract<
    DocumentGenerationStage,
    "queued" | "generating" | "saving"
  >;
  progress: number;
}

export interface DocumentGenerationDoneEvent {
  type: "document_done";
  generation_job_id: string;
  document_id: string;
  mode: Exclude<DocMode, "live">;
}

export interface DocumentGenerationErrorEvent {
  type: "document_error";
  generation_job_id: string;
  document_id: string;
  code: string;
  message: string;
  retry_after: number | null;
}

export type DocumentGenerationEvent =
  | DocumentGenerationStatusEvent
  | DocumentGenerationDoneEvent
  | DocumentGenerationErrorEvent;
