"""WavFrontend — compute log-Mel filterbank features from raw audio."""

from __future__ import annotations

import numpy as np

from meetasr.runpod.register import tables


def compute_lfr_features(fbank_feats: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """Stack Fbank features according to FunASR LFR specifications.

    Args:
        fbank_feats: Fbank feature matrix with shape [T, D].
        lfr_m: Number of adjacent frames to stack.
        lfr_n: Frame stride between stacked windows.

    Returns:
        LFR feature matrix with shape [ceil(T / lfr_n), D * lfr_m].

    Raises:
        ValueError: If the input shape or LFR parameters are invalid.
    """
    if fbank_feats.ndim != 2:
        raise ValueError(f"Expected 2-D fbank features, got shape {fbank_feats.shape}")
    if lfr_m <= 0 or lfr_n <= 0:
        raise ValueError(f"lfr_m and lfr_n must be positive, got {lfr_m=} {lfr_n=}")

    frame_count, feature_dim = fbank_feats.shape
    if frame_count == 0:
        return np.empty((0, feature_dim * lfr_m), dtype=np.float32)

    stacked_feats = []
    radius = (lfr_m - 1) // 2

    for center in range(0, frame_count, lfr_n):
        frame_indices = [
            max(0, min(index, frame_count - 1))
            for index in range(center - radius, center - radius + lfr_m)
        ]
        stacked = np.concatenate([fbank_feats[index] for index in frame_indices], axis=0)
        stacked_feats.append(stacked)

    return np.asarray(stacked_feats, dtype=np.float32)


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
        lfr_m: int = 7,
        lfr_n: int = 6,
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
        return compute_lfr_features(feats, lfr_m=self.lfr_m, lfr_n=self.lfr_n)
