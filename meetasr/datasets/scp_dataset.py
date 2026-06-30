"""SCP Dataset loader for Kaldi/FunASR formatted wav.scp files."""

import logging
from typing import Any, Dict, Iterator, List, Tuple, Optional
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    # Fallback if torch is not installed
    Dataset = object
    HAS_TORCH = False

from meetasr.utils.audio import load_audio, SAMPLE_RATE

logger = logging.getLogger(__name__)


class SCPDataset(Dataset):
    """Dataset loader reading from a wav.scp file.
    
    Format of wav.scp:
    <utt_id> <path_to_wav_file>
    """

    def __init__(
        self, 
        scp_path: str, 
        target_sr: int = SAMPLE_RATE, 
        frontend: Optional[Any] = None
    ):
        """Initialize SCPDataset.

        Args:
            scp_path: Path to the wav.scp file.
            target_sr: Target sample rate for audio.
            frontend: Optional WavFrontend instance to extract features.
        """
        self.scp_path = scp_path
        self.target_sr = target_sr
        self.frontend = frontend
        self.data: List[Tuple[str, str]] = []
        self._load_scp()

    def _load_scp(self) -> None:
        """Parse the scp file."""
        try:
            with open(self.scp_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        self.data.append((parts[0], parts[1]))
            logger.info(f"Loaded {len(self.data)} entries from {self.scp_path}")
        except Exception as e:
            logger.error(f"Failed to read scp file {self.scp_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item by index.
        
        Returns:
            Dict containing:
                - key: utterance ID (str)
                - speech: audio array or Fbank features (np.ndarray)
                - sample_rate: target sample rate
        """
        utt_id, audio_path = self.data[idx]
        try:
            waveform = load_audio(audio_path, target_sr=self.target_sr)
            
            # Extract features if frontend is provided
            if self.frontend is not None:
                speech = self.frontend.forward(waveform)
            else:
                speech = waveform
                
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path} for ID {utt_id}: {e}")
            speech = np.zeros(0, dtype=np.float32)

        return {
            "key": utt_id,
            "speech": speech,
            "sample_rate": self.target_sr
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Allow standard Python iteration."""
        for i in range(len(self)):
            yield self[i]


def collate_fn_speech(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate samples of varying lengths into a padded batch.
    
    Supports fallback to NumPy if PyTorch is not available.
    """
    keys = [item["key"] for item in batch]
    
    if HAS_TORCH:
        # PyTorch implementation
        speeches = [torch.tensor(item["speech"]) for item in batch]
        lengths = torch.tensor([s.size(0) for s in speeches], dtype=torch.int32)
        padded_speeches = torch.nn.utils.rnn.pad_sequence(
            speeches, batch_first=True, padding_value=0.0
        )
        return {
            "keys": keys,
            "speech": padded_speeches,         # [Batch, Max_T, Feat_Dim]
            "speech_lengths": lengths          # [Batch]
        }
    else:
        # NumPy fallback implementation
        speeches = [item["speech"] for item in batch]
        lengths = np.array([s.shape[0] for s in speeches], dtype=np.int32)
        max_len = lengths.max()
        
        # Determine shape (1D for audio, 2D for features)
        if speeches[0].ndim == 1:
            padded_speeches = np.zeros((len(batch), max_len), dtype=np.float32)
            for i, s in enumerate(speeches):
                padded_speeches[i, :s.shape[0]] = s
        else:
            feat_dim = speeches[0].shape[1]
            padded_speeches = np.zeros((len(batch), max_len, feat_dim), dtype=np.float32)
            for i, s in enumerate(speeches):
                padded_speeches[i, :s.shape[0], :] = s
                
        return {
            "keys": keys,
            "speech": padded_speeches,
            "speech_lengths": lengths
        }
