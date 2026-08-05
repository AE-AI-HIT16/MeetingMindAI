"""Dataset loader SCP cho các tệp wav.scp theo chuẩn Kaldi/FunASR."""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

#thử nạp thư viện PyTorch.
# Nếu không có sẵn thì sẽ tự động chuyển sang chế độ dự phòng dùng numpy
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    Dataset = object
    HAS_TORCH = False

from meetasr.utils.audio import SAMPLE_RATE, load_audio

logger = logging.getLogger(__name__)


class SCPDataset(Dataset):
    """Trình nạp dữ liệu đọc từ tệp danh sách cấu hình wav.scp.
    
    Định dạng chuẩn của tệp wav.scp:
    <mã_định_danh_câu_thoại> <đường_dẫn_tới_tệp_âm_thanh>
    """

    def __init__(
        self,
        scp_path: str,
        target_sr: int = SAMPLE_RATE,
        frontend: Optional[Any] = None
    ):
        """Khởi tạo đối tượng SCPDataset.

        Args:
            scp_path: Đường dẫn tuyệt đối hoặc tương đối tới tệp wav.scp.
            target_sr: Tần số lấy mẫu mục tiêu (chuẩn hóa âm thanh đầu vào).
            frontend: (Tùy chọn) Đối tượng WavFrontend dùng để trích xuất đặc trưng âm thanh (ví dụ: Fbank).
        """
        self.scp_path = scp_path
        self.target_sr = target_sr
        self.frontend = frontend
        # Kho chứa danh sách các cặp (ID, Đường dẫn)
        self.data: List[Tuple[str, str]] = []
        self._load_scp()

    def _load_scp(self) -> None:
        """Đọc và phân tích cú pháp tệp scp để nạp danh sách đường dẫn vào bộ nhớ."""
        try:
            with open(self.scp_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Loại bỏ khoảng trắng thừa hoặc ký tự xuống dòng
                    line = line.strip()

                    # Bỏ qua các dòng trống rỗng
                    if not line:
                        continue

                    # Tách dòng thành 2 phần: Mã ID và Đường dẫn (chỉ cắt ở khoảng trắng đầu tiên)
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        self.data.append((parts[0], parts[1]))

            logger.info(f"Đã nạp thành công {len(self.data)} bản ghi từ {self.scp_path}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc tệp scp {self.scp_path}: {e}")
            raise

    def __len__(self) -> int:
        """Trả về tổng số lượng bản ghi âm thanh có trong tập dữ liệu."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Truy xuất một mẫu dữ liệu dựa trên chỉ mục (index).
        
        Returns:
            Một từ điển (Dict) bao gồm:
                - key: Mã định danh câu thoại (chuỗi - str).
                - speech: Ma trận âm thanh thô hoặc đặc trưng Fbank (np.ndarray).
                - sample_rate: Tần số lấy mẫu hệ thống đang sử dụng.
        """
        utt_id, audio_path = self.data[idx]
        try:
            # Tải âm thanh và tự động ép về tần số lấy mẫu mục tiêu (16kHz)
            waveform = load_audio(audio_path, target_sr=self.target_sr)

            # Nếu có bộ tiền xử lý (frontend), thực hiện trích xuất đặc trưng ngay lập tức
            if self.frontend is not None:
                speech = self.frontend.forward(waveform)
            else:
                speech = waveform

        except Exception as e:
            # Ghi nhận lỗi nhưng không làm sập luồng, trả về mảng rỗng để tiếp tục xử lý
            logger.warning(f"Không thể tải tệp âm thanh {audio_path} cho ID {utt_id}: {e}")
            speech = np.zeros(0, dtype=np.float32)

        return {
            "key": utt_id,
            "speech": speech,
            "sample_rate": self.target_sr
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Biến đối tượng thành trình vòng lặp (iterator) chuẩn của Python."""
        for i in range(len(self)):
            yield self[i]


def collate_fn_speech(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Nhóm các mẫu âm thanh/đặc trưng có độ dài khác nhau thành một mẻ (batch) đồng nhất.
    
    Hàm này tự động thêm giá trị 0 (padding) vào cuối các mẫu ngắn để tất cả
    có cùng độ dài với mẫu dài nhất trong mẻ.
    Hỗ trợ linh hoạt cả PyTorch và NumPy (nếu PyTorch không được cài đặt).
    """
    keys = [item["key"] for item in batch]

    if HAS_TORCH:

        # Triển khai bằng PyTorch (Tối ưu cho việc huấn luyện GPU)
        speeches = [torch.tensor(item["speech"]) for item in batch]
        # Ghi nhận lại độ dài thực tế của từng chuỗi trước khi padding
        lengths = torch.tensor([s.size(0) for s in speeches], dtype=torch.int32)

        # Hàm pad_sequence sẽ tự động đệm số 0 vào các tensor ngắn
        padded_speeches = torch.nn.utils.rnn.pad_sequence(
            speeches, batch_first=True, padding_value=0.0
        )
        return {
            "keys": keys,
            "speech": padded_speeches,         # Hình dạng: [Batch, Max_T, Feat_Dim]
            "speech_lengths": lengths          # Hình dạng: [Batch]
        }
    else:
        # Triển khai dự phòng bằng NumPy (Dành cho môi trường API)

        speeches = [item["speech"] for item in batch]
        lengths = np.array([s.shape[0] for s in speeches], dtype=np.int32)
        max_len = lengths.max()

        # Xác định hình dạng mảng: 1D cho âm thanh thô, 2D cho đặc trưng Fbank
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
