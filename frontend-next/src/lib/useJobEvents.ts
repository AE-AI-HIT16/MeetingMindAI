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

function jobEventsUrl(jobId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
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

    const connect = () => {
      socket = new WebSocket(jobEventsUrl(jobId));

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
          const incoming = mapTranscriptSegment(event.segment);
          setSegments((current) => {
            const index = current.findIndex((item) =>
              incoming.id !== null
                ? item.id === incoming.id
                : item.startMs === incoming.startMs &&
                  item.endMs === incoming.endMs &&
                  item.text === incoming.text,
            );
            if (index < 0) return [...current, incoming];
            const next = [...current];
            next[index] = incoming;
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

        if (event.type === "done") {
          terminal = true;
          setDone(true);
          setStage("done");
          setProgress(1);
          setLiveDocumentId(event.live_document_id);
          socket?.close();
          return;
        }

        terminal = true;
        setError(event.message);
        socket?.close();
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
