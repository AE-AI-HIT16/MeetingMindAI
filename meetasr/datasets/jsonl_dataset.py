"""Dataset loader cho tệp JSONL chứa metadata và transcription."""

import json
import logging
from typing import Any, Dict, Iterator, List

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object

logger = logging.getLogger(__name__)


class JSONLDataset(Dataset):
    """Đọc và quản lý dữ liệu từ tệp định dạng JSON Lines .jsonl
    
    Định dạng yêu cầu: Mỗi dòng trong tệp phải là một chuỗi JSON hợp lệ và độc lập.
    """

    def __init__(self, jsonl_path: str):
        """Khởi tạo 
        Args:
            jsonl_path: Đường dẫn tuyệt đối hoặc tương đối tới tệp .jsonl.
        """
        self.jsonl_path = jsonl_path
        # Kho chứa toàn bộ các đối tượng JSON trên RAM
        self.data: List[Dict[str, Any]] = []
        self._load_jsonl()

    def _load_jsonl(self) -> None:
        """Đọc tệp jsonl và phân tích cú pháp từng dòng thành đối tượng Dictionary."""
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        # Chuyển đổi chuỗi văn bản thành Python Dictionary
                        obj = json.loads(line)
                        self.data.append(obj)
                    except json.JSONDecodeError as e:
                        # Ghi nhận cảnh báo nếu có dòng lỗi cú pháp thay vì dừng toàn bộ quá trình nạp
                        logger.warning(f"Không thể phân tích cú pháp dòng {line_idx+1} trong {self.jsonl_path}: {e}")

            logger.info(f"Đã nạp thành công {len(self.data)} đối tượng JSON từ {self.jsonl_path}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc tệp JSONL {self.jsonl_path}: {e}")
            raise

    def __len__(self) -> int:
        """Trả về tổng số lượng bản ghi JSON đã được nạp thành công-> xác định kích thức tập dữ liệu
        
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Truy xuất một bản ghi dữ liệu (Dictionary) cụ thể dựa trên chỉ mục (index)."""
        return self.data[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Biến đối tượng thành một trình vòng lặp (iterator) chuẩn của Python.
        
        Cho phép sử dụng cú pháp: `for item in dataset:`
        """
        for i in range(len(self)):
            yield self[i]
