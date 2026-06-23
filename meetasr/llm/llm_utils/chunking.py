from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

import tiktoken

MAX_TOKEN = 1500
OVERLAP_SENTENCES = 3

enc = tiktoken.encoding_for_model("gpt-4o-mini")

# =========================
# 1. BUILD WINDOWS
# =========================
def build_windows(sentence_info: List[Dict], window_size: int = 3):
    windows = []
    n = len(sentence_info)

    for i in range(n - window_size + 1):
        window = sentence_info[i:i + window_size]

        merged_text = " ".join(
            f"{u.get('speaker') or 'UNKNOWN'}: {u.get('text','')}"
            for u in window
        )

        windows.append({
            "start_idx": i,
            "end_idx": i + window_size,
            "start_time": window[0]["start"],
            "end_time": window[-1]["end"],
            "merged_text": merged_text
        })

    return windows


# =========================
# 2. EMBEDDINGS
# =========================
def windows_to_embeddings(windows: List[Dict], model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [w["merged_text"] for w in windows]
    return model.encode(texts)

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# =========================
# 3. COSINE SIM
# =========================
def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a, b)


# 4. SMOOTH SIM (optional but useful)
def smooth_similarities(embeddings, alpha=0.7):
    sims = []

    for i in range(1, len(embeddings)):
        sim = cosine_sim(embeddings[i], embeddings[i - 1])

        if i == 1:
            sims.append(sim)
        else:
            sims.append(alpha * sim + (1 - alpha) * sims[-1])

    return sims

# 5. ADAPTIVE THRESHOLD
def adaptive_threshold(similarities):
    return np.mean(similarities) - 0.5 * np.std(similarities)


# 6. DETECT BOUNDARIES
def detect_boundaries(embeddings, threshold=0.75):
    boundaries = [0]

    for i in range(1, len(embeddings)):
        sim = cosine_sim(embeddings[i], embeddings[i - 1])

        if sim < threshold:
            boundaries.append(i)

    boundaries.append(len(embeddings))
    return sorted(list(set(boundaries)))


# CONVERT SENTENCES → MARKDOWN
def sentence_info_to_markdown(sentences: List[Dict]) -> str:
    lines = []

    for s in sentences:
        speaker = s.get("speaker") or "UNKNOWN"
        start = s.get("start", 0.0)
        text = s.get("text", "").strip()

        if not text:
            continue

        lines.append(f"[{start:.2f}] {speaker}: {text}")

    return "\n".join(lines)


# BUILD CHUNKS (FIXED LOGIC)
def build_chunks_from_boundaries(sentence_info: List[Dict], boundaries: List[int]):
    chunks = []
    n = len(sentence_info)

    i = 0
    chunk_index = 0

    while i < n:
        # lấy overlap từ chunk trước
        overlap_start = max(0, i - OVERLAP_SENTENCES)
        current_sentences = sentence_info[overlap_start:i]

        start_i = i

        while i < n:
            candidate_sentences = sentence_info[overlap_start:i + 1]

            text = sentence_info_to_markdown(candidate_sentences)
            token_count = count_tokens(text)

            # nếu vượt token thì dừng chunk
            if token_count > MAX_TOKEN:
                break

            i += 1

        # nếu không tiến được (1 câu quá dài)
        if start_i == i:
            i += 1
            continue

        chunk_sentences = sentence_info[overlap_start:i]

        chunks.append({
            "chunk_index": chunk_index,
            "start_time": chunk_sentences[0]["start"],
            "end_time": chunk_sentences[-1]["end"],
            "utterance_count": len(chunk_sentences),
            "markdown": sentence_info_to_markdown(chunk_sentences)
        })

        chunk_index += 1

    return chunks

# FINAL PIPELINE WRAPPER
def run_chunking_pipeline(sentence_info: List[Dict],
                          window_size=3,
                          threshold=0.75):

    windows = build_windows(sentence_info, window_size)
    embeddings = windows_to_embeddings(windows)

    boundaries = detect_boundaries(embeddings, threshold)

    chunks = build_chunks_from_boundaries(sentence_info, boundaries)

    return chunks


# OPTIONAL: COMPRESS OUTPUT (JSON-LIKE)
def convert_chunks_to_markdown(chunks: List[Dict]) -> str:
    """
    Compact format for LLM (low token usage)
    """

    out = []

    for c in chunks:
        out.append(
            f"## CHUNK {c['chunk_index']} "
            f"[{c['start_time']:.2f} → {c['end_time']:.2f}]\n"
            f"{c['markdown']}\n"
            f"---"
        )

    return "\n".join(out)

def run_pipeline(sentence_info: List[Dict],
                 window_size: int = 3,
                 threshold: float = 0.75,
                 use_adaptive_threshold: bool = False,
                 model_name: str = "all-MiniLM-L6-v2"):

    # 1. BUILD WINDOWS
    windows = build_windows(sentence_info, window_size)

    # 2. EMBEDDINGS
    embeddings = windows_to_embeddings(windows, model_name)

    # 3. OPTIONAL: adaptive threshold
    if use_adaptive_threshold:
        similarities = smooth_similarities(embeddings)
        threshold = adaptive_threshold(similarities)

    # 4. DETECT BOUNDARIES
    boundaries = detect_boundaries(embeddings, threshold)

    # 5. BUILD CHUNKS
    chunks = build_chunks_from_boundaries(sentence_info, boundaries)

    return chunks

def run_pipeline_with_markdown(sentence_info: List[Dict],
                               window_size: int = 3,
                               threshold: float = 0.75,
                               use_adaptive_threshold: bool = False,
                               model_name: str = "all-MiniLM-L6-v2"):

    chunks = run_pipeline(
        sentence_info,
        window_size,
        threshold,
        use_adaptive_threshold,
        model_name
    )

    markdown = convert_chunks_to_markdown(chunks)

    return chunks, markdown