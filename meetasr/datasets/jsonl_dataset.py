"""JSONL Dataset loader for metadata and text transcriptions."""

import json
import logging
from typing import Any, Dict, Iterator, List

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object

logger = logging.getLogger(__name__)


class JSONLDataset(Dataset):
    """Load metadata from a JSON Lines (.jsonl) file.
    
    Format: Each line is a valid JSON object.
    """

    def __init__(self, jsonl_path: str):
        """Initialize JSONLDataset.

        Args:
            jsonl_path: Path to the .jsonl file.
        """
        self.jsonl_path = jsonl_path
        self.data: List[Dict[str, Any]] = []
        self._load_jsonl()

    def _load_jsonl(self) -> None:
        """Parse the jsonl file."""
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self.data.append(obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_idx+1} in {self.jsonl_path}: {e}")
            logger.info(f"Loaded {len(self.data)} JSON objects from {self.jsonl_path}")
        except Exception as e:
            logger.error(f"Failed to read JSONL file {self.jsonl_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get the parsed JSON dictionary by index."""
        return self.data[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Allow standard Python iteration."""
        for i in range(len(self)):
            yield self[i]
