"use client";

import { useEffect, useState } from "react";
import type {
  DocumentGenerationEvent,
  DocumentGenerationStage,
  DocumentGenerationStatus,
} from "./types";

export interface DocumentGenerationEventsState {
  isConnected: boolean;
  status: DocumentGenerationStatus | "idle";
  stage: DocumentGenerationStage | null;
  progress: number;
  documentId: string | null;
  error: string | null;
}

interface TrackedGenerationState extends DocumentGenerationEventsState {
  generationJobId: string;
}

function documentEventsUrl(generationJobId: string): string {
  const envHost = process.env.NEXT_PUBLIC_WS_HOST || process.env.NEXT_PUBLIC_MEETASR_API || process.env.MEETASR_API || "http://56.10.9.132:8000";

  let raw = envHost.trim();
  if (!/^https?:\/\//i.test(raw) && !/^wss?:\/\//i.test(raw)) {
    raw = `http://${raw}`;
  }

  try {
    const parsed = new URL(raw);
    const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
    const wsScheme = isHttps ? "wss:" : "ws:";
    return `${wsScheme}//${parsed.host}/v1/document-jobs/${generationJobId}/events`;
  } catch (err) {
    console.error("Failed to parse WS URL for document events:", envHost, err);
    return `ws://56.10.9.132:8000/v1/document-jobs/${generationJobId}/events`;
  }
}

export function useDocumentGenerationEvents(
  generationJobId: string | null,
): DocumentGenerationEventsState {
  const [tracked, setTracked] = useState<TrackedGenerationState | null>(
    null,
  );

  useEffect(() => {
    if (!generationJobId) return;

    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    let terminal = false;

    const connect = () => {
      socket = new WebSocket(documentEventsUrl(generationJobId));
      socket.onopen = () => {
        setTracked((current) => ({
          generationJobId,
          isConnected: true,
          status:
            current?.generationJobId === generationJobId
              ? current.status
              : "queued",
          stage:
            current?.generationJobId === generationJobId
              ? current.stage
              : "queued",
          progress:
            current?.generationJobId === generationJobId
              ? current.progress
              : 0,
          documentId:
            current?.generationJobId === generationJobId
              ? current.documentId
              : null,
          error: null,
        }));
      };
      socket.onmessage = (message) => {
        let event: DocumentGenerationEvent;
        try {
          event = JSON.parse(String(message.data)) as DocumentGenerationEvent;
        } catch {
          return;
        }

        if (event.type === "document_status") {
          setTracked({
            generationJobId,
            isConnected: true,
            status:
              event.stage === "queued" ? "queued" : "processing",
            stage: event.stage,
            progress: event.progress,
            documentId: event.document_id,
            error: null,
          });
          return;
        }

        terminal = true;
        if (event.type === "document_done") {
          setTracked({
            generationJobId,
            isConnected: true,
            status: "done",
            stage: "done",
            progress: 1,
            documentId: event.document_id,
            error: null,
          });
        } else {
          setTracked({
            generationJobId,
            isConnected: true,
            status: "failed",
            stage: "failed",
            progress: 0,
            documentId: event.document_id || null,
            error: event.message,
          });
        }
        socket?.close();
      };
      socket.onerror = () => {
        // A temporary WebSocket failure is not a generation failure. Keep the
        // controls locked and reconnect to the persisted snapshot.
        setTracked((current) =>
          current?.generationJobId === generationJobId
            ? { ...current, isConnected: false }
            : current,
        );
      };
      socket.onclose = () => {
        setTracked((current) =>
          current?.generationJobId === generationJobId
            ? { ...current, isConnected: false }
            : current,
        );
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
  }, [generationJobId]);

  if (!generationJobId) {
    return {
      isConnected: false,
      status: "idle",
      stage: null,
      progress: 0,
      documentId: null,
      error: null,
    };
  }
  if (tracked?.generationJobId !== generationJobId) {
    return {
      isConnected: false,
      status: "queued",
      stage: "queued",
      progress: 0,
      documentId: null,
      error: null,
    };
  }
  return tracked;
}
