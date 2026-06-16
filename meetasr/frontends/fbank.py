"""WavFrontend — compute log-Mel filterbank features from raw audio."""

from __future__ import annotations

import numpy as np

from meetasr.register import tables


@tables.register("frontend_classes", key="WavFrontend")
class WavFrontend:
    """Compute log-Mel filterbank (FBANK) features.

    Compatible with Paraformer and SenseVoice model expectations.
    Output shape: [T, n_mels] as float32 numpy array.

    Args:
        fs: Sample rate (default 16000).
        n_mels: Number of mel filterbanks (default 80).
        frame_length: Frame length in ms (default 25).
        frame_shift: Frame shift in ms (default 10).
        dither: Dither coefficient (default 0.0 for inference).
        lfr_m: LFR m factor for Paraformer (default 7).
        lfr_n: LFR n factor for Paraformer (default 6).
    """

    def __init__(
        self,
        fs: int = 16000,
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.0,
        lfr_m: int = 1,
        lfr_n: int = 1,
        **kwargs,
    ):
        self.fs = fs
        self.n_mels = n_mels
        self.frame_length_ms = frame_length
        self.frame_shift_ms = frame_shift
        self.dither = dither
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n

    def output_size(self) -> int:
        """Return feature dimension size."""
        return self.n_mels * self.lfr_m

    def forward(self, audio: np.ndarray) -> np.ndarray:
        """Extract FBANK features from audio.

        Args:
            audio: Float32 mono audio at self.fs Hz. Shape [N].

        Returns:
            FBANK features of shape [T, n_mels] or [T, n_mels * lfr_m].
        """
        import librosa

        frame_len = int(self.fs * self.frame_length_ms / 1000)
        hop_len = int(self.fs * self.frame_shift_ms / 1000)

        # Log-Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.fs,
            n_fft=frame_len,
            hop_length=hop_len,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.fs // 2,
            power=2.0,
        )
        log_mel = np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)  # [T, n_mels]

        # LFR (Low Frame Rate) — Paraformer specific
        if self.lfr_m > 1 or self.lfr_n > 1:
            log_mel = self._apply_lfr(log_mel)

        return log_mel

    def _apply_lfr(self, feats: np.ndarray) -> np.ndarray:
        """Apply Low Frame Rate stacking/skipping."""
        T, D = feats.shape
        T_lfr = (T - self.lfr_m) // self.lfr_n + 1
        lfr_feats = np.zeros((T_lfr, D * self.lfr_m), dtype=np.float32)
        for i in range(T_lfr):
            start = i * self.lfr_n
            end = min(start + self.lfr_m, T)
            chunk = feats[start:end]
            # Pad if last chunk is shorter
            if chunk.shape[0] < self.lfr_m:
                pad = np.tile(feats[-1:], (self.lfr_m - chunk.shape[0], 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            lfr_feats[i] = chunk.flatten()
        return lfr_feats
