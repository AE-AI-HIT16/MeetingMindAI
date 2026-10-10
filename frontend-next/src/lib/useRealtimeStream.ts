"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getSession } from "next-auth/react";
import { mapTranscriptSegment } from "./mappers";
import { reduceRealtimeTranscripts } from "./realtimeTranscriptState";
import type {
  RealtimeTranscriptType,
  TranscriptDelta,
} from "./realtimeTranscriptState";
import type { ApiTranscriptSegment } from "./types";
import { getWebSocketBase } from "./runtime";
import { assertSessionHasToken } from "./api";

export type { SentenceInfo, TranscriptDelta } from "./realtimeTranscriptState";

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------

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
  /** Last error (message string or original error), if any. */
  error: unknown;
  /** Source ID created by backend for this realtime session. */
  sourceId: string | null;
  /** Offline (diarization) job ID; it starts processing right after stop. */
  jobId: string | null;
  /** Recording timeline (ms) already processed by server ASR. */
  confirmedEndMs: number;
}

export interface RealtimeStreamOptions {
  /** Server ASR window length in seconds (server clamps to 5–30). */
  windowSeconds?: number;
  /** Keywords / names passed to Qwen3-ASR as context. */
  context?: string;
  /** Known number of speakers for the offline diarization after stop. */
  speakers?: number;
}

export interface RealtimeStreamActions {
  /** Resolves once streaming starts (null on failure). */
  start: (
    source: AudioSource,
    options?: RealtimeStreamOptions,
  ) => Promise<{ startedAt: number; audioTrack: MediaStreamTrack } | null>;
  stop: () => void;
}

// ----------------------------------------------------------------
// Constants
// ----------------------------------------------------------------

/** Build the WebSocket URL based on the unified API base URL. */
async function buildWsUrl(options: RealtimeStreamOptions): Promise<string> {
  const params = new URLSearchParams();
  if (options.windowSeconds) params.set("window", String(options.windowSeconds));
  if (options.context?.trim()) params.set("context", options.context.trim());
  if (options.speakers) params.set("speakers", String(options.speakers));
  const query = params.toString();
  return `${getWebSocketBase()}/v1/realtime/stream${query ? `?${query}` : ""}`;
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
  const [error, setError] = useState<unknown>(null);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [confirmedEndMs, setConfirmedEndMs] = useState(0);

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
      audioCtxRef.current.close().catch(() => { });
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
  const start = useCallback(async (source: AudioSource, options: RealtimeStreamOptions = {}) => {
    setError(null);
    setTranscripts([]);
    setSourceId(null);
    setJobId(null);
    setConfirmedEndMs(0);
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
        throw new Error("Tab bạn chọn chưa chia sẻ âm thanh. Khi chọn tab, hãy bật mục “Chia sẻ âm thanh của tab”.");
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
      const wsUrl = await buildWsUrl(options);
      const authSession = await getSession();
      assertSessionHasToken(authSession);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";

      await new Promise<void>((resolve, reject) => {
        let settled = false;
        ws.onopen = () => {
          if (!settled) {
            settled = true;
            ws.send(
              JSON.stringify({
                type: "auth",
                token: authSession?.accessToken ?? null,
              }),
            );
            setIsConnected(true);
            resolve();
          }
        };
        ws.onerror = (ev) => {
          if (!settled) {
            settled = true;
            console.error("[RealtimeStream] WebSocket onerror event triggered:", ev);
            console.error("[RealtimeStream] Attempted WS URL:", wsUrl);
            console.error("[RealtimeStream] Current NEXT_PUBLIC_MEETASR_API:", process.env.NEXT_PUBLIC_MEETASR_API);
            reject(new Error("Không kết nối được tới máy chủ ghi âm. Vui lòng thử lại sau giây lát."));
          }
        };
        // Timeout after 5 s
        setTimeout(() => {
          if (!settled) {
            settled = true;
            reject(new Error("Máy chủ ghi âm phản hồi quá chậm. Vui lòng thử lại sau giây lát."));
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
            setJobId(data.job_id ?? null);
            return;
          }

          if (data.type === "transcript_confirmed") {
            setConfirmedEndMs((prev) => Math.max(prev, Number(data.end_ms) || 0));
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
            setError("Chưa hoàn tất được phần ghi âm cuối. Phần đã ghi vẫn được lưu và xử lý.");
            cleanup();
            return;
          }

          if (
            (data.type === "transcript_delta" ||
              data.type === "transcript_partial") &&
            segment
          ) {
            const mapped = mapTranscriptSegment(segment);
            setTranscripts((prev) =>
              reduceRealtimeTranscripts(prev, {
                type: data.type as RealtimeTranscriptType,
                receivedAt: Date.now(),
                segment: mapped,
              }),
            );
          }
        } catch {
          // ignore non-JSON
        }
      };

      ws.onclose = (ev) => {
        // Session already finished (stream_stopped detached it): whatever close
        // code follows (e.g. 1006 from the proxy) is not an error.
        if (wsRef.current !== ws) {
          setIsConnected(false);
          return;
        }
        if (finalizationTimerRef.current) {
          clearTimeout(finalizationTimerRef.current);
          finalizationTimerRef.current = null;
        }
        if (finalizingRef.current) {
          setError("Kết nối bị ngắt trước khi hoàn tất phần cuối. Phần đã ghi vẫn được lưu và xử lý.");
        } else if (ev.code !== 1000 && ev.code !== 1005) {
          // Abnormal close while recording (e.g. 1013 = realtime pipeline not
          // ready on the server). Previously this closed silently.
          setError(
            ev.code === 1013
              ? "Máy chủ đang khởi động. Vui lòng thử lại sau giây lát."
              : "Mất kết nối tới máy chủ trong lúc ghi. Phần đã ghi vẫn được lưu; hãy kiểm tra mạng rồi bấm ghi lại.",
          );
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
        setError("Kết nối tới máy chủ bị gián đoạn. Vui lòng kiểm tra mạng rồi bấm ghi lại.");
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
      return { startedAt: startTimeRef.current, audioTrack: stream.getAudioTracks()[0] };
    } catch (err: unknown) {
      // Keep the original error so the UI can explain it (mic permission,
      // no microphone, mic busy, connection…) — see friendlyError().
      setError(err ?? "Chưa bắt đầu ghi âm được. Vui lòng thử lại.");
      cleanup();
      return null;
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
      setError("Mất kết nối trước khi kết thúc ghi âm. Phần đã ghi vẫn được lưu và xử lý.");
      cleanup();
      return;
    }

    finalizingRef.current = true;
    setIsFinalizing(true);
    ws.send(JSON.stringify({ type: "stop" }));
    finalizationTimerRef.current = setTimeout(() => {
      setError("Máy chủ xử lý phần cuối lâu hơn bình thường. Bản ghi vẫn được lưu, bạn có thể xem lại trong thư viện.");
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
    jobId,
    confirmedEndMs,
    start,
    stop,
  };
}
