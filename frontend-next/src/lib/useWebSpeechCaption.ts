"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  EMPTY_CAPTION_STATE,
  reduceLiveCaption,
  settleInterim,
  type LiveCaptionState,
} from "./liveCaptionState";

export type CaptionStatus = "idle" | "listening" | "restarting" | "error";

/**
 * Web Speech works in Chrome/Edge. Brave exposes webkitSpeechRecognition but
 * every session fails with "network" (no Google speech key), so treat it as
 * unsupported. Firefox has no implementation.
 */
export function isWebSpeechSupported(): boolean {
  if (typeof window === "undefined") return false;
  if (navigator.brave) return false;
  return Boolean(window.SpeechRecognition ?? window.webkitSpeechRecognition);
}

/** Errors after which captions give up; the page falls back to server-only. */
/**
 * Chrome 135+ desktop accepts a MediaStreamTrack in start() (captions for tab
 * audio). There is no feature detection for it (WebAudio/web-speech-api#126),
 * so gate on the browser version; older versions would silently use the mic.
 */
export function isTrackCaptionSupported(): boolean {
  if (!isWebSpeechSupported()) return false;
  const chrome = navigator.userAgentData?.brands.find((b) => b.brand === "Google Chrome");
  return Boolean(chrome && Number(chrome.version) >= 135);
}

const FATAL_ERRORS = new Set([
  "not-allowed",
  "service-not-allowed",
  "audio-capture",
  "network",
  "language-not-supported",
]);

/** Chrome ends sessions after silence / ~1 min; give up only on a restart storm. */
const MAX_RESTARTS_PER_WINDOW = 8;
const RESTART_WINDOW_MS = 10_000;

export function useWebSpeechCaption(lang = "vi-VN") {
  const recRef = useRef<SpeechRecognition | null>(null);
  const wantRunningRef = useRef(false);
  const startedAtRef = useRef(0);
  const trackRef = useRef<MediaStreamTrack | undefined>(undefined);
  const deliveredFinalsRef = useRef(0);
  const restartTimesRef = useRef<number[]>([]);

  const [state, setState] = useState<LiveCaptionState>(EMPTY_CAPTION_STATE);
  const [status, setStatus] = useState<CaptionStatus>("idle");

  const stop = useCallback(() => {
    wantRunningRef.current = false;
    const rec = recRef.current;
    recRef.current = null;
    if (rec) {
      rec.onend = null;
      rec.onerror = null;
      rec.onresult = null;
      try {
        rec.stop();
      } catch {
        // already stopped
      }
    }
    // Keep what the user already saw (interim becomes a line).
    setState(settleInterim);
    setStatus((prev) => (prev === "error" ? prev : "idle"));
  }, []);

  const start = useCallback(
    (startedAt: number, track?: MediaStreamTrack) => {
      const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
      if (!Ctor) return;

      stop();
      startedAtRef.current = startedAt;
      trackRef.current = track;
      restartTimesRef.current = [];
      wantRunningRef.current = true;
      setState(EMPTY_CAPTION_STATE);

      const rec = new Ctor();
      rec.lang = lang;
      rec.continuous = true;
      rec.interimResults = true;
      rec.maxAlternatives = 1;
      recRef.current = rec;

      rec.onstart = () => {
        deliveredFinalsRef.current = 0; // results list resets per session
        setStatus("listening");
      };

      rec.onresult = (ev) => {
        const finals: string[] = [];
        let interim = "";
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const result = ev.results[i];
          if (result.isFinal) {
            if (i >= deliveredFinalsRef.current) {
              finals.push(result[0].transcript);
              deliveredFinalsRef.current = i + 1;
            }
          } else {
            interim += result[0].transcript;
          }
        }
        const atMs = Date.now() - startedAtRef.current;
        setState((prev) => reduceLiveCaption(prev, { finals, interim, atMs }));
      };

      rec.onerror = (ev) => {
        if (FATAL_ERRORS.has(ev.error)) {
          wantRunningRef.current = false;
          setStatus("error");
        }
        // "no-speech" / "aborted": onend fires next and restarts.
      };

      rec.onend = () => {
        if (!wantRunningRef.current || recRef.current !== rec) return;
        const now = Date.now();
        const recent = restartTimesRef.current.filter((t) => now - t < RESTART_WINDOW_MS);
        recent.push(now);
        restartTimesRef.current = recent;
        if (recent.length > MAX_RESTARTS_PER_WINDOW) {
          wantRunningRef.current = false;
          setStatus("error");
          return;
        }
        setStatus("restarting");
        setState(settleInterim);
        setTimeout(() => {
          if (!wantRunningRef.current || recRef.current !== rec) return;
          try {
            rec.start(trackRef.current);
          } catch {
            // InvalidStateError: already started / track ended
          }
        }, 250);
      };

      try {
        rec.start(trackRef.current);
      } catch {
        wantRunningRef.current = false;
        setStatus("error");
      }
    },
    [lang, stop],
  );

  useEffect(() => stop, [stop]);

  return {
    status,
    lines: state.lines,
    interim: state.interim,
    start,
    stop,
  };
}
