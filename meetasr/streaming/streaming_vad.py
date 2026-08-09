"""Stateful streaming VAD endpointing independent of any model backend."""

from __future__ import annotations

import numpy as np

from meetasr.streaming.streaming_vad_types import (
    SpeechProbabilityPredictor,
    SpeechUtterance,
    StreamingVADConfig,
    UtteranceReason,
)


class StreamingVAD:
    """Convert frame probabilities into lossless, endpointed speech windows."""

    def __init__(
        self,
        predictor: SpeechProbabilityPredictor,
        config: StreamingVADConfig | None = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or StreamingVADConfig()
        self._frame_buffer = np.empty(0, dtype=np.float32)
        self._processed_samples = 0
        self._speaking = False
        self._speech_run_samples = 0
        self._speech_run_start = 0
        self._silence_samples = 0
        self._idle_audio = np.empty(0, dtype=np.float32)
        self._idle_start_sample = 0
        self._utterance_parts: list[np.ndarray] = []
        self._utterance_start_sample = 0
        self._utterance_samples = 0

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def processed_ms(self) -> int:
        return self._processed_samples * 1000 // self.config.sample_rate

    @property
    def current_utterance_start_ms(self) -> int | None:
        """Absolute start of the active VAD utterance, including pre-roll."""
        if not self._speaking:
            return None
        return self._utterance_start_sample * 1000 // self.config.sample_rate

    @property
    def current_utterance_duration_ms(self) -> int:
        """Audio accumulated for the active utterance, including pre-roll."""
        start_ms = self.current_utterance_start_ms
        if start_ms is None:
            return 0
        return self.processed_ms - start_ms

    def process(self, audio: np.ndarray) -> list[SpeechUtterance]:
        """Consume arbitrary-sized float32 audio and return completed windows."""
        self._validate_audio(audio)
        if audio.size == 0:
            return []

        self._frame_buffer = np.concatenate((self._frame_buffer, audio))
        utterances: list[SpeechUtterance] = []
        frame_samples = self.config.frame_samples
        while self._frame_buffer.size >= frame_samples:
            frame = self._frame_buffer[:frame_samples].copy()
            self._frame_buffer = self._frame_buffer[frame_samples:].copy()
            utterances.extend(self._process_frame(frame, frame))
        return utterances

    def flush(self) -> list[SpeechUtterance]:
        """Finalize a partial frame and any in-progress short utterance."""
        utterances: list[SpeechUtterance] = []
        if self._frame_buffer.size:
            actual = self._frame_buffer.copy()
            padded = np.pad(
                actual,
                (0, self.config.frame_samples - actual.size),
            ).astype(np.float32, copy=False)
            self._frame_buffer = self._frame_buffer[:0]
            utterances.extend(self._process_frame(actual, padded))

        if self._speaking:
            speech_end = self._processed_samples - self._silence_samples
            end_sample = min(
                self._processed_samples,
                speech_end + self.config.samples_for_ms(
                    self.config.post_roll_ms
                ),
            )
            item = self._emit(end_sample, "session_stop", keep_speaking=False)
            if item is not None:
                utterances.append(item)
        elif self._speech_run_samples > 0 and self._idle_audio.size:
            start_sample = max(
                self._idle_start_sample,
                self._speech_run_start
                - self.config.samples_for_ms(self.config.pre_roll_ms),
            )
            offset = start_sample - self._idle_start_sample
            audio = self._idle_audio[offset:].copy()
            if audio.size:
                utterances.append(
                    SpeechUtterance(
                        audio=audio,
                        start_sample=start_sample,
                        end_sample=start_sample + audio.size,
                        sample_rate=self.config.sample_rate,
                        reason="session_stop",
                    )
                )

        self.predictor.reset()
        self._reset_buffers()
        return utterances

    def _process_frame(
        self,
        audio: np.ndarray,
        predictor_frame: np.ndarray,
    ) -> list[SpeechUtterance]:
        frame_start = self._processed_samples
        frame_end = frame_start + audio.size
        probability = float(self.predictor.predict(predictor_frame))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("VAD probability must be between 0 and 1")
        self._processed_samples = frame_end

        if not self._speaking:
            self._append_idle(audio, frame_start)
            if probability >= self.config.start_threshold:
                if self._speech_run_samples == 0:
                    self._speech_run_start = frame_start
                self._speech_run_samples += audio.size
            else:
                self._speech_run_samples = 0

            if self._speech_run_samples >= self.config.samples_for_ms(
                self.config.min_speech_ms
            ):
                self._begin_utterance()
            return []

        self._utterance_parts.append(audio)
        self._utterance_samples += audio.size
        if probability < self.config.end_threshold:
            self._silence_samples += audio.size
        else:
            self._silence_samples = 0

        if self._silence_samples >= self.config.samples_for_ms(
            self.config.min_silence_ms
        ):
            speech_end = frame_end - self._silence_samples
            end_sample = min(
                frame_end,
                speech_end
                + self.config.samples_for_ms(self.config.post_roll_ms),
            )
            item = self._emit(end_sample, "speech_end", keep_speaking=False)
            return [item] if item is not None else []

        max_samples = self.config.samples_for_ms(
            self.config.max_utterance_ms
        )
        if self._utterance_samples >= max_samples:
            item = self._emit(
                self._utterance_start_sample + max_samples,
                "max_window",
                keep_speaking=True,
            )
            return [item] if item is not None else []
        return []

    def _append_idle(self, audio: np.ndarray, start_sample: int) -> None:
        if self._idle_audio.size == 0:
            self._idle_start_sample = start_sample
        self._idle_audio = np.concatenate((self._idle_audio, audio))
        keep_samples = (
            self.config.samples_for_ms(
                self.config.pre_roll_ms + self.config.min_speech_ms
            )
            + self.config.frame_samples
        )
        if self._idle_audio.size > keep_samples:
            discarded = self._idle_audio.size - keep_samples
            self._idle_audio = self._idle_audio[discarded:].copy()
            self._idle_start_sample += discarded

    def _begin_utterance(self) -> None:
        start_sample = max(
            self._idle_start_sample,
            self._speech_run_start
            - self.config.samples_for_ms(self.config.pre_roll_ms),
        )
        offset = start_sample - self._idle_start_sample
        audio = self._idle_audio[offset:].copy()
        self._speaking = True
        self._utterance_start_sample = start_sample
        self._utterance_parts = [audio]
        self._utterance_samples = audio.size
        self._idle_audio = self._idle_audio[:0]
        self._speech_run_samples = 0
        self._silence_samples = 0

    def _emit(
        self,
        end_sample: int,
        reason: UtteranceReason,
        *,
        keep_speaking: bool,
    ) -> SpeechUtterance | None:
        combined = (
            np.concatenate(self._utterance_parts)
            if self._utterance_parts
            else np.empty(0, dtype=np.float32)
        )
        emit_samples = max(
            0,
            min(combined.size, end_sample - self._utterance_start_sample),
        )
        emitted = combined[:emit_samples].copy()
        remainder = combined[emit_samples:].copy()
        actual_end = self._utterance_start_sample + emit_samples
        item = None
        if emitted.size:
            item = SpeechUtterance(
                audio=emitted,
                start_sample=self._utterance_start_sample,
                end_sample=actual_end,
                sample_rate=self.config.sample_rate,
                reason=reason,
            )

        if keep_speaking:
            self._utterance_start_sample = actual_end
            self._utterance_parts = [remainder] if remainder.size else []
            self._utterance_samples = remainder.size
        else:
            self._speaking = False
            self._utterance_parts = []
            self._utterance_samples = 0
            self._silence_samples = 0
            self._speech_run_samples = 0
            self._idle_audio = remainder
            self._idle_start_sample = actual_end
            self._trim_idle_audio()
        return item

    def _trim_idle_audio(self) -> None:
        keep_samples = self.config.samples_for_ms(self.config.pre_roll_ms)
        if self._idle_audio.size > keep_samples:
            discarded = self._idle_audio.size - keep_samples
            self._idle_audio = self._idle_audio[discarded:].copy()
            self._idle_start_sample += discarded

    def _reset_buffers(self) -> None:
        self._frame_buffer = self._frame_buffer[:0]
        self._speaking = False
        self._speech_run_samples = 0
        self._silence_samples = 0
        self._idle_audio = self._idle_audio[:0]
        self._utterance_parts = []
        self._utterance_samples = 0

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> None:
        if not isinstance(audio, np.ndarray):
            raise ValueError("audio must be a numpy.ndarray")
        if audio.ndim != 1 or audio.dtype != np.float32:
            raise ValueError("audio must be mono float32")
