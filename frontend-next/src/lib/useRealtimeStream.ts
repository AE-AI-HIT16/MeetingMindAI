"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getSession } from "next-auth/react";
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
  /** Whether the server is draining accepted audio after stop. */
  isFinalizing: boolean;
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

/** Build the WebSocket URL based on the current page location. */
function buildWsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  // In dev, the backend runs on port 8000; in prod, behind the same host.
  // We connect directly to the backend since Next.js rewrites don't proxy WS.
  const host = process.env.NEXT_PUBLIC_WS_HOST ?? "127.0.0.1:8000";
  return `${proto}//${host}/v1/realtime/stream`;
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
  const finalizationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startTimeRef = useRef<number>(0);
  const acceptAudioRef = useRef(false);
  const finalizingRef = useRef(false);

  // --- State ---
  const [isRecording, setIsRecording] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptDelta[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [sourceId, setSourceId] = useState<string | null>(null);

  // ------------------------------------------------------------------
  // Cleanup helper
  // ------------------------------------------------------------------
  const stopCapture = useCallback(() => {
    acceptAudioRef.current = false;

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
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    // Stop mic tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    setIsRecording(false);
  }, []);

  const cleanup = useCallback(() => {
    stopCapture();

    if (finalizationTimerRef.current) {
      clearTimeout(finalizationTimerRef.current);
      finalizationTimerRef.current = null;
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

    finalizingRef.current = false;
    setIsFinalizing(false);
    setIsConnected(false);
  }, [stopCapture]);

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
    finalizingRef.current = false;
    setIsFinalizing(false);

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
      const authSession = await getSession();
      const ws = new WebSocket(buildWsUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => {
          ws.send(
            JSON.stringify({
              type: "auth",
              token: authSession?.accessToken ?? null,
            }),
          );
          setIsConnected(true);
          resolve();
        };
        ws.onerror = () => {
          reject(new Error("Không thể kết nối WebSocket tới backend."));
        };
        // Timeout after 5 s
        setTimeout(
          () => reject(new Error("WebSocket connection timed out.")),
          5000,
        );
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

          if (data.type === "stream_stopping") {
            finalizingRef.current = true;
            setIsFinalizing(true);
            return;
          }

          if (data.type === "stream_stopped") {
            if (finalizationTimerRef.current) {
              clearTimeout(finalizationTimerRef.current);
              finalizationTimerRef.current = null;
            }
            finalizingRef.current = false;
            setIsFinalizing(false);
            setIsConnected(false);
            wsRef.current = null;
            ws.close(1000);
            return;
          }

          if (
            data.type === "error" &&
            (data.code === "stream_drain_failed" ||
              data.code === "stream_finalization_failed")
          ) {
            setError(data.message ?? "Không thể hoàn tất audio realtime.");
            cleanup();
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

              // A session has one active preview. Its start can move when the
              // backend bounds ASR work to a rolling audio window, so replace
              // the previous preview regardless of its old start timestamp.
              const next = [
                ...prev.filter((t) => t.type !== "transcript_partial"),
                item,
              ];

              next.sort((a, b) => a.startMs - b.startMs);
              return next;
            });
          }
        } catch {
          // ignore non-JSON
        }
      };

      ws.onclose = () => {
        if (finalizationTimerRef.current) {
          clearTimeout(finalizationTimerRef.current);
          finalizationTimerRef.current = null;
        }
        if (finalizingRef.current) {
          setError("Kết nối đóng trước khi server hoàn tất audio.");
        }
        finalizingRef.current = false;
        setIsFinalizing(false);
        setIsConnected(false);
        stopCapture();
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
      };

      ws.onerror = () => {
        setError("WebSocket error — kết nối bị gián đoạn.");
        cleanup();
      };

      // 5. Wire: mic → worklet → WS
      workletNode.port.onmessage = (ev: MessageEvent) => {
        const pcm16: ArrayBuffer = ev.data;
        if (acceptAudioRef.current && ws.readyState === WebSocket.OPEN) {
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
      acceptAudioRef.current = true;
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Lỗi không xác định khi ghi âm.";
      setError(msg);
      cleanup();
    }
  }, [cleanup, stopCapture]);

  // ------------------------------------------------------------------
  // STOP
  // ------------------------------------------------------------------
  const stop = useCallback(() => {
    if (!isRecording || finalizingRef.current) {
      return;
    }

    const ws = wsRef.current;
    stopCapture();

    if (ws?.readyState !== WebSocket.OPEN) {
      setError("WebSocket đã ngắt nên không thể xác nhận audio cuối.");
      cleanup();
      return;
    }

    finalizingRef.current = true;
    setIsFinalizing(true);
    ws.send(JSON.stringify({ type: "stop" }));
    finalizationTimerRef.current = setTimeout(() => {
      setError("Server mất quá nhiều thời gian để hoàn tất audio.");
      cleanup();
    }, 135_000);
  }, [cleanup, isRecording, stopCapture]);

  return {
    isRecording,
    isFinalizing,
    isConnected,
    transcripts,
    elapsedMs,
    error,
    sourceId,
    start,
    stop,
  };
}
