"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------

export interface TranscriptDelta {
  text: string;
  sentenceInfo: SentenceInfo[];
  /** Local timestamp (ms since epoch) when we received this delta. */
  receivedAt: number;
}

export interface SentenceInfo {
  text: string;
  start: number;
  end: number;
  speaker: string | number | null;
}

export type AudioSource = "microphone" | "tab";

interface TranscriptResultPayload {
  text?: string;
  sentence_info?: SentenceInfo[];
}

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
  const startTimeRef = useRef<number>(0);

  // --- State ---
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptDelta[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);

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
      audioCtxRef.current.close().catch(() => {});
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
      const ws = new WebSocket(buildWsUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => {
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
          const result = data.result as TranscriptResultPayload | undefined;
          if (data.type === "transcript_delta" && result?.text) {
            setTranscripts((prev) => [
              ...prev,
              { text: result.text, sentenceInfo: result.sentence_info ?? [], receivedAt: Date.now() },
            ]);
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
    start,
    stop,
  };
}
