from typing import List, Dict
from sentence_transformers import SentenceTransformer

# gom các câu 3 nhóm 1 thành 1 window
# VD: (
# W1 = {U1, U2, U3}
# W2 = {U2, U3, U4}
# )
# Sau đó embedding các window thaành vector
# Xem có window nào có độ tương đồng chênh lệch
# quá lớn với đứa đằng trước thi lấy đó là điểm cắt cho chunk
# Chuyển các chunk thành dạng json đơn giản để giảm token
# Rồi ghép các chunk nhỏ lại thành 1 chunk cho sao cho đảm bảo kích thước không vượt tối đa
# => Ta chunking thành công

def build_windows(data: List[Dict], window_size: int = 3):
    windows = []

    n = len(data)

    for i in range(n - window_size + 1):
        window = data[i:i + window_size]

        merged_text = " ".join(
            f"{u['speaker']}: {u['text']}"
            for u in window
        )

        windows.append({
            "start_time": window[0]["start_time"],
            "end_time": window[-1]["end_time"],
            "window_index": i,
            "merged_text": merged_text
        })

    return windows



def windows_to_embeddings(windows: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    texts = [w["merged_text"] for w in windows]

    embeddings = model.encode(texts)

    return embeddings

# Chuyển chunks có format notion thành dạng json đơn giản, giúp AI giảm token
def convert_chunks_to_json():
    pass