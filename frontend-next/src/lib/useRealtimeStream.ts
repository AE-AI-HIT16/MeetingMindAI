"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { mapTranscriptSegment } from "./mappers";
import type { ApiTranscriptSegment } from "./types";

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------

export interface TranscriptDelta {
  text: string;
  sentenceInfo: SentenceInfo[];
  receivedAt: number;

  startMs: number;
  endMs: number;
  speaker: number | null;

  // thêm để biết loại transcript
  type: "transcript_delta" | "transcript_partial";
}

export interface SentenceInfo {
  text: string;
  start: number;
  end: number;
  speaker: string | number | null;
}

export type AudioSource = "microphone" | "tab";

export interface RealtimeStreamState {
  /** Whether the mic is currently recording + streaming. */
  isRecording: boolean;
  /** Whether the WebSocket is connected and healthy. */
  isConnected: boolean;
  /** Ordered list of transcript deltas received from the backend. */
  transcripts: TranscriptDelta[];
  /** Elapsed recording time in milliseconds. */
  elapsedMs: number;
  /** Last error message, if any. */
  error: string | null;
  /** Source ID created by backend for this realtime session. */
  sourceId: string | null;
}

export interface RealtimeStreamActions {
  start: (source: AudioSource) => Promise<void>;
  stop: () => void;
}

// ----------------------------------------------------------------
// Constants
// ----------------------------------------------------------------

/** Build the WebSocket URL based on the unified API base URL. */
async function buildWsUrl(): Promise<string> {
  const envHost = process.env.NEXT_PUBLIC_WS_HOST || process.env.NEXT_PUBLIC_MEETASR_API || process.env.MEETASR_API || "http://56.10.9.132:8000";

  let raw = envHost.trim();
  if (!/^https?:\/\//i.test(raw) && !/^wss?:\/\//i.test(raw)) {
    raw = `http://${raw}`;
  }

  try {
    const parsed = new URL(raw);
    const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
    const wsScheme = isHttps ? "wss:" : "ws:";
    return `${wsScheme}//${parsed.host}/v1/realtime/stream`;
  } catch (err) {
    console.error("Failed to parse WS URL from:", envHost, err);
    const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
    const wsScheme = isHttps ? "wss:" : "ws:";
    return `${wsScheme}//56.10.9.132:8000/v1/realtime/stream`;
  }
}

// ----------------------------------------------------------------
// Hook
// ----------------------------------------------------------------

export function useRealtimeStream(): RealtimeStreamState &
  RealtimeStreamActions {
  // --- Refs (mutable across renders, no re-render triggers) ---
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  // --- State ---
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptDelta[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [sourceId, setSourceId] = useState<string | null>(null);

  // ------------------------------------------------------------------
  // Cleanup helper
  // ------------------------------------------------------------------
  const cleanup = useCallback(() => {
    // Stop timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Disconnect AudioWorklet
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    // Close AudioContext
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => { });
      audioCtxRef.current = null;
    }

    // Stop mic tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }

    setIsRecording(false);
    setIsConnected(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => cleanup, [cleanup]);

  // ------------------------------------------------------------------
  // START
  // ------------------------------------------------------------------
  const start = useCallback(async (source: AudioSource) => {
    setError(null);
    setTranscripts([]);
    setSourceId(null);
    setElapsedMs(0);

    try {
      const stream = source === "microphone"
        ? await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true },
        })
        : await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });

      if (stream.getAudioTracks().length === 0) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("Tab duoc chon khong chia se am thanh. Hay bat chia se am thanh khi chon tab.");
      }
      streamRef.current = stream;

      // 2. AudioContext + Worklet
      const audioCtx = new AudioContext({ sampleRate: undefined as unknown as number });
      audioCtxRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule("/pcm-processor.js");

      const workletNode = new AudioWorkletNode(audioCtx, "pcm-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
      });
      workletNodeRef.current = workletNode;

      // Tell processor the real sample rate
      workletNode.port.postMessage({
        command: "init",
        sampleRate: audioCtx.sampleRate,
      });

      // 3. Open WebSocket
      const wsUrl = await buildWsUrl();
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      await new Promise<void>((resolve, reject) => {
        let settled = false;
        ws.onopen = () => {
          if (!settled) {
            settled = true;
            setIsConnected(true);
            resolve();
          }
        };
        ws.onerror = (e) => {
          if (!settled) {
            settled = true;
            console.error("WebSocket connection error:", wsUrl, e);
            reject(new Error(`Không thể kết nối WebSocket tới backend (${wsUrl}).`));
          }
        };
        // Timeout after 5 s
        setTimeout(() => {
          if (!settled) {
            settled = true;
            reject(new Error(`WebSocket connection timed out (${wsUrl}).`));
          }
        }, 5000);
      });

      // 4. WS message handler — receives transcript_delta JSON
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(
            typeof ev.data === "string"
              ? ev.data
              : new TextDecoder().decode(ev.data),
          );
          const segment = data.segment as ApiTranscriptSegment | undefined;
          console.log("WS message:", data.type, segment);

          if (data.type === "session_init") {
            setSourceId(data.source_id ?? null);
            return;
          }

          if (data.type === "transcript_delta" && segment) {
            const mapped = mapTranscriptSegment(segment);

            setTranscripts((prev) => {
              const item: TranscriptDelta = {
                text: mapped.text,
                sentenceInfo: [
                  {
                    text: mapped.text,
                    start: mapped.startMs / 1000,
                    end: mapped.endMs / 1000,
                    speaker: mapped.speaker,
                  },
                ],
                receivedAt: Date.now(),
                startMs: mapped.startMs,
                endMs: mapped.endMs,
                speaker: mapped.speaker,
                type: "transcript_delta",
              };

              // Find if delta already exists for this exact startMs
              const deltaIndex = prev.findIndex(
                (t) => t.type === "transcript_delta" && t.startMs === mapped.startMs,
              );

              let next: TranscriptDelta[];
              if (deltaIndex !== -1) {
                next = [...prev];
                next[deltaIndex] = item;
              } else {
                next = [...prev, item];
              }

              // Remove only partial segments whose startMs <= this delta's endMs.
              // These partials are now confirmed/superseded by the delta.
              // Partials that start AFTER the delta are still in-progress and should remain.
              next = next.filter(
                (t) => t.type !== "transcript_partial" || t.startMs > mapped.endMs,
              );

              next.sort((a, b) => a.startMs - b.startMs);
              return next;
            });
          }

          if (data.type === "transcript_partial" && segment) {
            const mapped = mapTranscriptSegment(segment);

            setTranscripts((prev) => {
              // Get max endMs of confirmed transcript_delta segments
              const maxDeltaEndMs = prev.reduce(
                (max, t) =>
                  t.type === "transcript_delta" ? Math.max(max, t.endMs) : max,
                0,
              );

              // Ignore stale partials belonging to an already confirmed delta timeframe
              if (mapped.startMs < maxDeltaEndMs) {
                return prev;
              }

              const item: TranscriptDelta = {
                text: mapped.text,
                sentenceInfo: [
                  {
                    text: mapped.text,
                    start: mapped.startMs / 1000,
                    end: mapped.endMs / 1000,
                    speaker: mapped.speaker,
                  },
                ],
                receivedAt: Date.now(),
                startMs: mapped.startMs,
                endMs: mapped.endMs,
                speaker: mapped.speaker,
                type: "transcript_partial",
              };

              // Append partial: if a partial with the same startMs already exists
              // (same utterance being refined by ASR), update it in place.
              // Otherwise, add the new partial alongside existing ones so multiple
              // concurrent partial utterances are visible on the UI.
              const existingIdx = prev.findIndex(
                (t) => t.type === "transcript_partial" && t.startMs === mapped.startMs,
              );

              let next: TranscriptDelta[];
              if (existingIdx !== -1) {
                next = [...prev];
                next[existingIdx] = item;
              } else {
                next = [...prev, item];
              }
              //               const next = [...prev, item];

              next.sort((a, b) => a.startMs - b.startMs);
              return next;
            });
          }
        } catch {
          // ignore non-JSON
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      ws.onerror = () => {
        setError("WebSocket error — kết nối bị gián đoạn.");
        cleanup();
      };

      // 5. Wire: mic → worklet → WS
      workletNode.port.onmessage = (ev: MessageEvent) => {
        const pcm16: ArrayBuffer = ev.data;
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(pcm16);
        }
      };

      const mediaSource = audioCtx.createMediaStreamSource(stream);
      mediaSource.connect(workletNode);

      // 6. Timer
      startTimeRef.current = Date.now();
      timerRef.current = setInterval(() => {
        setElapsedMs(Date.now() - startTimeRef.current);
      }, 200);

      setIsRecording(true);
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Lỗi không xác định khi ghi âm.";
      setError(msg);
      cleanup();
    }
  }, [cleanup]);

  // ------------------------------------------------------------------
  // STOP
  // ------------------------------------------------------------------
  const stop = useCallback(() => {
    cleanup();
  }, [cleanup]);

  return {
    isRecording,
    isConnected,
    transcripts,
    elapsedMs,
    error,
    sourceId,
    start,
    stop,
  };
}
