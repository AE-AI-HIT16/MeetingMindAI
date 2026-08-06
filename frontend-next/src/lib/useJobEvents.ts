"use client";

import { useEffect, useState } from "react";
import { mapTranscriptSegment } from "./mappers";
import type {
  DocSection,
  JobEvent,
  ProcessingStage,
  TranscriptSegment,
} from "./types";

export interface JobEventsState {
  isConnected: boolean;
  stage: ProcessingStage;
  progress: number;
  segments: TranscriptSegment[];
  sections: DocSection[];
  done: boolean;
  liveDocumentId: string | null;
  error: string | null;
}

async function jobEventsUrl(jobId: string): Promise<string> {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  try {
    const res = await fetch("/api/ws-info");
    const data = await res.json();
    if (data.host && data.token) {
      return `wss://${data.host}/v1/jobs/${jobId}/events?api_key=${data.token}`;
    }
  } catch (e) {
    // Fallback
  }
  const host = process.env.NEXT_PUBLIC_WS_HOST ?? "127.0.0.1:8000";
  return `${protocol}//${host}/v1/jobs/${jobId}/events`;
}

export function useJobEvents(jobId: string | null): JobEventsState {
  const [isConnected, setIsConnected] = useState(false);
  const [stage, setStage] = useState<ProcessingStage>("extracting_audio");
  const [progress, setProgress] = useState(0);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [sections, setSections] = useState<DocSection[]>([]);
  const [done, setDone] = useState(false);
  const [liveDocumentId, setLiveDocumentId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    let terminal = false;

    const connect = async () => {
      const url = await jobEventsUrl(jobId);
      if (stopped) return;
      socket = new WebSocket(url);

      socket.onopen = () => {
        setIsConnected(true);
        setError(null);
        setDone(false);
        setStage("extracting_audio");
        setProgress(0);
        setSegments([]);
        setSections([]);
        setLiveDocumentId(null);
      };
      socket.onmessage = (message) => {
        let event: JobEvent;
        try {
          event = JSON.parse(String(message.data)) as JobEvent;
        } catch {
          return;
        }

        if (event.type === "status") {
          setStage(event.stage);
          setProgress(event.progress);
          return;
        }

        if (event.type === "transcript_delta") {
          const incoming = { ...mapTranscriptSegment(event.segment), _partialType: "delta" as const };
          setSegments((current) => {
            const index = current.findIndex((item) =>
              incoming.id !== null
                ? item.id === incoming.id
                : item.startMs === incoming.startMs &&
                  item.endMs === incoming.endMs &&
                  item.text === incoming.text,
            );

            let next: typeof current;
            if (index < 0) {
              next = [...current, incoming];
            } else {
              next = [...current];
              next[index] = incoming;
            }

            // Remove only partial segments whose startMs <= this delta's endMs.
            // These partials are now confirmed/superseded by the delta.
            // Partials that start AFTER the delta are still in-progress and should remain.
            next = next.filter(
              (item) => item._partialType !== "partial" || item.startMs > incoming.endMs,
            );

            return next;
          });
          return;
        }

        if (event.type === "doc_delta") {
          setSections((current) => {
            const nextSection: DocSection = {
              id: event.section_id,
              heading: "Tài liệu trực tiếp",
              markdown: event.markdown,
            };
            const index = current.findIndex(
              (section) => section.id === event.section_id,
            );
            if (index < 0) return [...current, nextSection];
            const next = [...current];
            next[index] = nextSection;
            return next;
          });
          return;
        }

        if (event.type === "speaker_update") {
          const updates = new Map(
            event.updates.map((update) => [
              update.segment_id,
              update.speaker,
            ]),
          );
          setSegments((current) =>
            current.map((segment) =>
              segment.id !== null && updates.has(segment.id)
                ? { ...segment, speaker: updates.get(segment.id) ?? null }
                : segment,
            ),
          );
          return;
        }

        if (event.type === "transcript_partial") {
          const incoming = { ...mapTranscriptSegment(event.segment), _partialType: "partial" as const };
          setSegments((current) => {
            // Get max endMs of confirmed delta segments
            const maxDeltaEndMs = current.reduce(
              (max, item) =>
                item._partialType === "delta" ? Math.max(max, item.endMs) : max,
              0,
            );

            // Ignore stale partials belonging to an already confirmed delta timeframe
            if (incoming.startMs < maxDeltaEndMs) {
              return current;
            }

            // Append partial: if a partial with the same startMs exists
            // (same utterance being refined by ASR), update it in place.
            // Otherwise, add the new partial alongside existing ones.
            const index = current.findIndex(
              (item) => item._partialType === "partial" && item.startMs === incoming.startMs,
            );
            if (index < 0) return [...current, incoming];
            const next = [...current];
            next[index] = incoming;
            return next;
          });
          return;
        }

        if (event.type === "done") {
          terminal = true;
          setDone(true);
          setStage("done");
          setProgress(1);
          setLiveDocumentId(event.live_document_id);
          socket?.close();
          return;
        }

        if (event.type === "error") {
          terminal = true;
          setError(event.message);
          socket?.close();
          return;
        }
      };

      socket.onerror = () => {
        setError("Không thể kết nối tới luồng tiến trình.");
      };
      socket.onclose = () => {
        setIsConnected(false);
        if (!stopped && !terminal) {
          reconnectTimer = setTimeout(connect, 1500);
        }
      };
    };

    connect();
    return () => {
      stopped = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [jobId]);

  return {
    isConnected,
    stage,
    progress,
    segments,
    sections,
    done,
    liveDocumentId,
    error: jobId ? error : "Source này không có Job để theo dõi.",
  };
}
