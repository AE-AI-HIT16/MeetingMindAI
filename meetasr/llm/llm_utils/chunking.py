import logging
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

MAX_TOKEN = 1500
OVERLAP_SENTENCES = 3
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_token_encoder() -> Any | None:
    """Load tiktoken lazily so importing this module never requires network."""
    try:
        import tiktoken

        return tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as exc:
        logger.warning(
            "tiktoken encoder is unavailable; using an approximate token count: %s",
            exc,
        )
        return None

# =========================
# 1. BUILD WINDOWS
# =========================
def build_windows(sentence_info: List[Dict], window_size: int = 3) -> List[Dict]:
    """Build overlapping windows from sentences.

    Args:
        sentence_info: List of sentence dictionaries containing speaker and text.
        window_size: Number of sentences in each window.

    Returns:
        A list of window dictionaries with merged text and timestamps.
    """
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
            "start_time": window[0].get("start", 0.0),
            "end_time": window[-1].get("end", 0.0),
            "merged_text": merged_text
        })

    return windows


# =========================
# 2. EMBEDDINGS
# =========================
def windows_to_embeddings(windows: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate sentence embeddings for sliding windows.

    Args:
        windows: List of window dictionaries.
        model_name: SentenceTransformer model name.

    Returns:
        Numpy array of embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Semantic chunking requires the optional 'legacy-llm' dependencies. "
            "Install with: pip install -e '.[legacy-llm]'"
        ) from exc

    model = SentenceTransformer(model_name)
    texts = [w["merged_text"] for w in windows]
    return np.array(model.encode(texts))

def count_tokens(text: str) -> int:
    encoder = _get_token_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    # Conservative local approximation used only when the optional tokenizer
    # is missing or its vocabulary cannot be downloaded.
    return max(1, (len(text) + 3) // 4)

# =========================
# 3. COSINE SIM
# =========================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity value.
    """
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def smooth_similarities(embeddings: np.ndarray, alpha: float = 0.7) -> List[float]:
    """Smooth similarities using Exponential Moving Average.

    Args:
        embeddings: Array of embeddings.
        alpha: Smoothing factor.

    Returns:
        List of smoothed similarity scores.
    """
    sims = []

    for i in range(1, len(embeddings)):
        sim = cosine_sim(embeddings[i], embeddings[i - 1])

        if i == 1:
            sims.append(sim)
        else:
            sims.append(alpha * sim + (1 - alpha) * sims[-1])

    return sims


def adaptive_threshold(similarities: List[float]) -> float:
    """Calculate an adaptive threshold based on mean and std deviation.

    Args:
        similarities: List of similarity scores.

    Returns:
        Calculated threshold value.
    """
    return float(np.mean(similarities) - 0.5 * np.std(similarities))


def detect_boundaries(embeddings: np.ndarray, threshold: float = 0.75) -> List[int]:
    """Detect topic boundaries based on embedding similarities.

    Args:
        embeddings: Array of embeddings.
        threshold: Boundary detection threshold.

    Returns:
        Sorted list of boundary indices.
    """
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
def build_chunks_from_boundaries(
    sentence_info: List[Dict],
    boundaries: List[int]
) -> List[Dict]:
    """Build text chunks from detected boundaries respecting max token limits.

    Args:
        sentence_info: List of sentence dictionaries.
        boundaries: List of boundary indices.

    Returns:
        List of generated chunks.
    """
    chunks = []
    n = len(sentence_info)

    i = 0
    chunk_index = 0

    while i < n:
        # lấy overlap từ chunk trước
        overlap_start = max(0, i - OVERLAP_SENTENCES)
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
            "start_time": chunk_sentences[0].get("start", 0.0),
            "end_time": chunk_sentences[-1].get("end", 0.0),
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

def run_pipeline(
    sentence_info: List[Dict],
    window_size: int = 3,
    threshold: float = 0.75,
    use_adaptive_threshold: bool = False,
    model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict]:
    """Run the complete chunking pipeline.

    Args:
        sentence_info: List of sentence dictionaries.
        window_size: Number of sentences per window.
        threshold: Cosine similarity threshold for boundary detection.
        use_adaptive_threshold: Not currently implemented.
        model_name: SentenceTransformer model name.

    Returns:
        List of chunk dictionaries.
    """
    windows = build_windows(sentence_info, window_size)
    embeddings = windows_to_embeddings(windows, model_name)
    boundaries = detect_boundaries(embeddings, threshold)
    chunks = build_chunks_from_boundaries(sentence_info, boundaries)

    logger.debug(f"Chunking: {len(sentence_info)} sentences -> {len(chunks)} chunks")
    return chunks

def run_pipeline_with_markdown(
    sentence_info: List[Dict],
    window_size: int = 3,
    threshold: float = 0.75,
    use_adaptive_threshold: bool = False,
    model_name: str = "all-MiniLM-L6-v2"
) -> tuple[List[Dict], str]:
    """Run chunking pipeline and format output as markdown.

    Args:
        sentence_info: List of sentence dictionaries.
        window_size: Number of sentences per window.
        threshold: Cosine similarity threshold.
        use_adaptive_threshold: Boolean flag.
        model_name: SentenceTransformer model name.

    Returns:
        A tuple containing the list of chunk dicts and the markdown string.
    """
    chunks = run_pipeline(
        sentence_info,
        window_size,
        threshold,
        use_adaptive_threshold,
        model_name
    )
    markdown = convert_chunks_to_markdown(chunks)
    return chunks, markdown
