# Document Planner — Thiết kế chi tiết cho AI Engineer 2

> Tài liệu triển khai đầy đủ cho hạn chế #9 trong
> [13_phase2_limitations_and_fixes.md](13_phase2_limitations_and_fixes.md): thay
> `MeetingSummarizer` (template cứng theo "cuộc họp") bằng **`DocumentPlanner`** — kiến trúc
> Plan → Write tổng quát cho **bất kỳ** loại audio/video nào (họp, bài giảng, podcast, phỏng vấn,
> hội thoại đời thường...), scale sang domain mới **mà không cần sửa code**.
>
> Đọc trước: mục 9 của doc 13 (nhận xét hiện trạng), và
> [11_phase2_notebooklm_plan.md](11_phase2_notebooklm_plan.md) mục 3.2 (lớp sinh docs realtime) —
> tài liệu này là bản thiết kế chi tiết cho đúng phần đó, viết chung với việc thay
> `MeetingSummarizer`.

---

## 1. Mục tiêu & phi-mục tiêu

**Mục tiêu:**
- Một pipeline tóm tắt/tài liệu hoá duy nhất, dùng được cho **mọi loại nội dung nói**, không cần
  biết trước domain là gì.
- Thêm khả năng xử lý một loại nội dung mới (vd: "review sản phẩm", "buổi tư vấn tâm lý") = **0
  dòng code, 0 file prompt mới**.
- Transcript dài (>1 giờ) phải thực sự chạy map-reduce, không nhét nguyên văn vào 1 lời gọi LLM.
- Dùng chung cho cả 2 luồng: **finalize** (Phase 2 `doc_gen.py`, đã có) và **realtime doc**
  (Phase 2 `worker.py` sự kiện `doc_delta`, đang là text thô — xem mục 3 doc 13).

**Phi-mục tiêu (ngoài phạm vi doc này):**
- Không đổi API/DB của Phase 1 (`MeetingReport`, route `/v1/meeting/summarize`). Giữ nguyên,
  chạy song song.
- Không tự động dịch ngôn ngữ — vẫn nhận `language` như hiện tại.
- Không làm real-time diarization tốt hơn — đó là việc của AI Engineer 1 (mục 1, 5 doc 13).

---

## 2. Kiến trúc tổng quan

```
                         ┌───────────────────────────────┐
                         │   TranscriptResult / segments  │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │   Bước A — PLAN (1 lần gọi)     │
                         │   plan_outline(sample, llm)     │
                         │   → content_kind + outline[]    │
                         └───────────────┬───────────────┘
                                         │  outline = [{id, heading, kind}, ...]
                         ┌───────────────▼───────────────┐
                         │   Bước B — WRITE (map-reduce    │
                         │   theo từng mục trong outline)  │
                         │   write_section(section, chunks)│
                         └───────────────┬───────────────┘
                                         │  list[DocSection]
                         ┌───────────────▼───────────────┐
                         │        DocumentReport          │
                         │  (id, content_kind, sections[], │
                         │   raw_transcript)               │
                         └─────────────────────────────────┘
```

**Nguyên tắc thiết kế cốt lõi:** không có khái niệm "domain" ở bất kỳ đâu trong code. Chỉ có
`content_kind` — một **chuỗi mô tả tự do do LLM sinh ra** để hiển thị cho người dùng (vd:
"Cuộc họp công việc", "Bài giảng"), và `outline` — danh sách mục **do LLM tự đề xuất** dựa trên
nội dung thực tế, không tra bảng if/else nào cả.

---

## 3. Cấu trúc dữ liệu mới

Tạo file `meetasr/schemas_doc.py` (tách riêng khỏi `schemas.py` để không đụng model Phase 1):

```python
"""Generic document schema — content-agnostic, replaces the meeting-only
MeetingReport for anything built on DocumentPlanner (Phase 2 finalize + realtime).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass
class DocSection:
    """One section of a generated document.

    `kind` is an open vocabulary (see SUGGESTED_KINDS below) — new values are
    expected and fine. Never validate `kind` against a closed enum; it exists
    only so the frontend can pick a consistent icon for known values and fall
    back to a generic one otherwise.
    """

    id: str
    heading: str
    kind: str            # e.g. "summary", "key_points", "action_items", or anything new
    markdown: str = ""


@dataclass
class DocumentReport:
    """Full output of DocumentPlanner: a content-agnostic structured document."""

    content_kind: str                       # free-text label, e.g. "Bài giảng"
    sections: list[DocSection] = field(default_factory=list)
    language: str = "vi"
    llm_model: str = ""
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = [f"# {self.content_kind}", ""]
        for sec in self.sections:
            lines += [f"## {sec.heading}", "", sec.markdown, ""]
        return "\n".join(lines).strip()
```

Đây chính là khuôn `DocSection` đã dùng ở `frontend-next/src/lib/types.ts` — chỉ thêm trường
`kind` (frontend hiện chưa có, thêm optional, không phá tương thích: `kind?: string`).

---

## 4. Prompt — chỉ 2 file, dùng cho MỌI loại nội dung

Xoá tư duy "1 file prompt / 1 domain / 1 tác vụ" (hiện có 8 file: `summarize|topics|action_items|
decisions` × `meeting|consultation`). Thay bằng **2 file duy nhất**, đặt tại
`meetasr/llm/prompts/plan_vi.txt` và `meetasr/llm/prompts/write_section_vi.txt` (+ hậu tố `_en.txt`
khi cần đa ngôn ngữ, theo đúng quy ước hiện có).

### 4.1 `plan_vi.txt` (Bước A)

```text
Bạn là trợ lý biên tập tài liệu chuyên nghiệp. Nhiệm vụ của bạn là ĐỌC đoạn trích transcript dưới
đây và ĐỀ XUẤT một bộ khung mục lục phù hợp nhất để biến nội dung này thành một tài liệu văn bản
hữu ích cho người đọc.

Đoạn trích transcript (có thể chỉ là một phần của nội dung dài hơn):
{transcript_sample}

QUAN TRỌNG — đây không phải biên bản cuộc họp mặc định. Nội dung có thể là: cuộc họp công việc,
bài giảng, podcast, phỏng vấn, buổi tư vấn, video hướng dẫn, hội thoại đời thường, hoặc bất kỳ
thể loại nào khác. Hãy tự nhận diện thể loại từ chính nội dung, KHÔNG giả định trước.

Hãy trả về JSON với cấu trúc sau. KHÔNG bao bọc trong markdown code block.

{{
  "content_kind": "Tên ngắn gọn mô tả loại nội dung này, bằng tiếng Việt tự nhiên (vd: 'Cuộc họp
                   công việc', 'Bài giảng học thuật', 'Cuộc trò chuyện phiếm', 'Buổi phỏng vấn')",
  "outline": [
    {{"id": "s1", "heading": "Tiêu đề mục", "kind": "nhãn_loại_mục"}},
    ...
  ]
}}

Nguyên tắc chọn mục lục:
- CHỈ đề xuất những mục THỰC SỰ có căn cứ trong nội dung. Ví dụ: nếu không có công việc nào được
  giao, ĐỪNG thêm mục kiểu "việc cần làm" — không tạo mục rỗng để cho đủ khuôn mẫu.
- Số mục hợp lý: 2-6 mục tuỳ độ phức tạm của nội dung. Nội dung ngắn/đơn giản có thể chỉ cần 2 mục.
- Trường "kind" là nhãn tự do, nhưng ưu tiên dùng lại các nhãn quen thuộc nếu phù hợp:
  summary, key_points, action_items, decisions, quotes, timeline, glossary, open_questions,
  people, resources. Nếu nội dung cần một loại mục khác không có trong danh sách này, cứ đặt
  nhãn "kind" mới mô tả đúng bản chất mục đó (vd: "trieu_chung", "cong_thuc", "y_tuong_hay").
- Mục đầu tiên luôn nên là một bản tóm tắt tổng quan ngắn ("kind": "summary").
- Sắp xếp mục lục theo trình tự đọc hợp lý (không nhất thiết theo thời gian trong transcript).

Nếu transcript quá ngắn hoặc không có nội dung thực chất, trả về outline chỉ với 1 mục "summary".
```

### 4.2 `write_section_vi.txt` (Bước B — map, mỗi chunk)

```text
Bạn đang viết nội dung cho một mục trong tài liệu tổng hợp từ transcript audio/video.

Tên tài liệu / loại nội dung: {content_kind}
Mục đang viết: "{heading}" (loại: {kind})

Đây là một phần transcript (có thể không phải toàn bộ — chỉ viết dựa trên phần này, không suy
diễn thêm):
{chunk}

Hãy viết nội dung markdown cho mục "{heading}" dựa CHỈ trên phần transcript trên. Yêu cầu:
- Nếu "kind" là "summary" hoặc "key_points": viết văn xuôi hoặc bullet ngắn gọn, không lặp lại
  nguyên văn transcript.
- Nếu "kind" là "action_items", "decisions", "quotes", "timeline" hoặc tương tự: dùng danh sách
  có cấu trúc (bullet hoặc bảng markdown) với đủ ngữ cảnh (ai, khi nào nếu có).
- Nếu phần transcript này KHÔNG có nội dung liên quan đến mục "{heading}", trả về chuỗi rỗng
  (không bịa nội dung để lấp đầy).
- Không thêm tiêu đề mục (heading) vào trong nội dung — chỉ trả phần thân.
```

### 4.3 Prompt gộp (Bước B — reduce, khi có nhiều chunk cho cùng 1 mục)

Dùng ngay `write_section_vi.txt` với `{chunk}` = nối các nháp từ bước map lại, kèm câu dẫn
`"Dưới đây là các bản nháp rời rạc cho cùng một mục, hãy gộp và viết lại thành một bản hoàn
chỉnh, mạch lạc, không lặp ý:"` — không cần file prompt thứ 3. Xem code `_reduce()` ở mục 6.

---

## 5. API `DocumentPlanner`

Tạo `meetasr/llm/planner.py`:

```python
"""DocumentPlanner — content-agnostic summarization via Plan → Write.

Replaces MeetingSummarizer for Phase 2 (see meet_docs/docs/14_document_planner_design.md).
MeetingSummarizer / MeetingReport are kept untouched for the Phase 1 API — this is a new,
parallel code path, not a modification of the old one.
"""

from __future__ import annotations

import json
import logging
import re
import time

from meetasr.llm.abs_llm import AbsLLMClient
from meetasr.schemas import SentenceInfo, TranscriptResult
from meetasr.schemas_doc import DocSection, DocumentReport

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 6000       # per map-step LLM call
SAMPLE_CHARS_FOR_PLAN = 4000     # how much transcript the planner itself sees


class DocumentPlanner:
    def __init__(
        self,
        client: AbsLLMClient,
        language: str = "vi",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.language = language
        self.temperature = temperature
        self.max_tokens = max_tokens
        from meetasr.llm.llm_utils.prompts import load_generic_prompts
        self._prompts = load_generic_prompts(language=language)  # {"plan": ..., "write_section": ...}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan_and_write(self, transcript: TranscriptResult) -> DocumentReport:
        """Full pipeline: plan outline, then map-reduce write each section."""
        t0 = time.perf_counter()
        full_text = self._format_transcript(transcript)

        content_kind, outline = self._plan(full_text)

        chunks = self._chunk(full_text, MAX_CHARS_PER_CHUNK)
        sections = [
            self._write_section(sec, content_kind, chunks) for sec in outline
        ]
        # Drop sections that came back empty (grounding rule: no filler content).
        sections = [s for s in sections if s.markdown.strip()]

        return DocumentReport(
            content_kind=content_kind,
            sections=sections,
            language=self.language,
            llm_model=getattr(self.client, "model", ""),
            processing_time=round(time.perf_counter() - t0, 2),
        )

    def plan_only(self, transcript_sample: str) -> tuple[str, list[dict]]:
        """Expose the plan step alone — used by the realtime doc stream (doc 13 §3)."""
        return self._plan(transcript_sample)

    def write_one_section(
        self, section: dict, content_kind: str, text: str
    ) -> DocSection:
        """Expose a single write call — used by the realtime doc stream per chunk."""
        return self._write_section(section, content_kind, [text])

    # ------------------------------------------------------------------
    # Step A — Plan
    # ------------------------------------------------------------------
    def _plan(self, full_text: str) -> tuple[str, list[dict]]:
        sample = self._representative_sample(full_text, SAMPLE_CHARS_FOR_PLAN)
        prompt = self._prompts["plan"].format(transcript_sample=sample)
        try:
            raw = self.client.chat(prompt, temperature=self.temperature, max_tokens=1024)
            data = _parse_json_object(raw)
            content_kind = data.get("content_kind", "Tài liệu")
            outline = data.get("outline", [])
            if not outline:
                raise ValueError("empty outline")
            return content_kind, outline
        except Exception as e:  # noqa: BLE001
            logger.warning("Plan step failed (%s) — falling back to single summary section.", e)
            return "Tài liệu", [{"id": "s1", "heading": "Tóm tắt", "kind": "summary"}]

    def _representative_sample(self, text: str, max_chars: int) -> str:
        """Head + middle + tail excerpt so the planner sees the whole shape of
        long content without paying for the full transcript."""
        if len(text) <= max_chars:
            return text
        third = max_chars // 3
        mid_start = len(text) // 2 - third // 2
        return (
            text[:third]
            + "\n...\n"
            + text[mid_start : mid_start + third]
            + "\n...\n"
            + text[-third:]
        )

    # ------------------------------------------------------------------
    # Step B — Write (map-reduce per section)
    # ------------------------------------------------------------------
    def _write_section(
        self, section: dict, content_kind: str, chunks: list[str]
    ) -> DocSection:
        drafts = [
            self._call_write(section, content_kind, chunk)
            for chunk in chunks
        ]
        drafts = [d for d in drafts if d.strip()]

        if not drafts:
            markdown = ""
        elif len(drafts) == 1:
            markdown = drafts[0]
        else:
            markdown = self._reduce(section, content_kind, drafts)

        return DocSection(
            id=section.get("id", "s"),
            heading=section.get("heading", ""),
            kind=section.get("kind", "summary"),
            markdown=markdown,
        )

    def _call_write(self, section: dict, content_kind: str, chunk: str) -> str:
        prompt = self._prompts["write_section"].format(
            content_kind=content_kind,
            heading=section.get("heading", ""),
            kind=section.get("kind", "summary"),
            chunk=chunk,
        )
        try:
            return self.client.chat(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            ).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Write step failed for section '%s': %s", section.get("heading"), e)
            return ""

    def _reduce(self, section: dict, content_kind: str, drafts: list[str]) -> str:
        """Merge multiple chunk-drafts of the same section into one coherent pass."""
        joined = "\n\n---\n\n".join(drafts)
        merge_note = (
            "\n\nDưới đây là các bản nháp rời rạc cho cùng một mục, hãy gộp và viết lại "
            "thành một bản hoàn chỉnh, mạch lạc, không lặp ý:\n\n" + joined
        )
        prompt = self._prompts["write_section"].format(
            content_kind=content_kind,
            heading=section.get("heading", ""),
            kind=section.get("kind", "summary"),
            chunk=merge_note,
        )
        try:
            return self.client.chat(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            ).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Reduce step failed for section '%s': %s", section.get("heading"), e)
            return joined  # degrade gracefully: concatenated drafts beat losing content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_transcript(self, result: TranscriptResult) -> str:
        if result.sentence_info:
            lines = []
            for s in result.sentence_info:
                spk = f"Speaker {s.speaker}: " if s.speaker is not None else ""
                lines.append(f"[{s.start:.1f}s] {spk}{s.text}")
            return "\n".join(lines)
        return result.text

    def _chunk(self, text: str, max_chars: int) -> list[str]:
        """Split on line boundaries so a sentence is never cut mid-way."""
        lines = text.split("\n")
        chunks, current = [], []
        size = 0
        for line in lines:
            if size + len(line) > max_chars and current:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += len(line)
        if current:
            chunks.append("\n".join(current))
        return chunks or [text]


def _parse_json_object(raw: str) -> dict:
    match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    text = match.group(1).strip() if match else raw.strip()
    if not match:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)
```

**Điểm cần chú ý khi cài đặt:**
- `_chunk()` cắt theo **ranh giới dòng** (mỗi dòng transcript = 1 câu có timestamp), không bao giờ
  cắt giữa câu — khắc phục luôn kiểu lỗi mà `_truncate()` no-op của `MeetingSummarizer` đang có nguy
  cơ gặp phải khi ai đó bật lại nó.
- Mỗi section được map-reduce **độc lập** với các section khác → có thể chạy `_write_section`
  song song bằng `concurrent.futures.ThreadPoolExecutor` nếu cần tối ưu tốc độ (không bắt buộc ở
  bản đầu, ghi chú ở mục 8).
- Section rỗng sau khi write bị loại bỏ (`sections = [s for s in sections if s.markdown.strip()]`)
  — đúng nguyên tắc "không bịa nội dung để lấp đầy" nêu trong prompt.

---

## 6. Loader prompt tổng quát

Thêm hàm mới vào `meetasr/llm/llm_utils/prompts.py` (không sửa `load_prompts` cũ — Phase 1 vẫn
dùng nó nguyên vẹn):

```python
def load_generic_prompts(language: str = "vi") -> dict[str, str]:
    """Load the 2 domain-agnostic prompts used by DocumentPlanner.

    Unlike load_prompts(), this has NO prompt_type/domain axis — that is the
    whole point (see meet_docs/docs/14_document_planner_design.md).
    """
    def _read(name: str) -> str:
        path = os.path.join(_PROMPT_DIR, f"{name}_{language}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prompt file: {path}")
        with open(path, encoding="utf-8") as f:
            return f.read()

    return {
        "plan": _read("plan"),
        "write_section": _read("write_section"),
    }
```

---

## 7. Nối vào Phase 2 (thay thế 2 chỗ dùng `MeetingSummarizer` kiểu cũ)

### 7.1 `meetasr/realtime/doc_gen.py` — luồng finalize

Thay `build_summary()` hiện tại (gọi thẳng `summarizer.summarize()` → `MeetingReport.to_markdown()`)
bằng việc gọi `DocumentPlanner`:

```python
def build_summary(title: str, segments: List[TranscriptSegment], planner) -> str:
    """Structured document via DocumentPlanner. Raises if no planner configured."""
    if planner is None:
        raise RuntimeError("No LLM planner configured (add an 'llm' block to config).")

    transcript = TranscriptResult(
        key=title,
        text=" ".join(s.text for s in segments),
        duration=(segments[-1].end_ms / 1000.0) if segments else 0.0,
        sentence_info=[
            SentenceInfo(text=s.text, start=s.start_ms / 1000.0, end=s.end_ms / 1000.0, speaker=s.speaker)
            for s in segments
        ],
    )
    report = planner.plan_and_write(transcript)
    return report.to_markdown()
```

Và trong `meetasr/api/routes/realtime.py`, đổi chỗ lấy `summarizer` thành lấy `planner`
(xem mục 9 để biết cách nạp `DocumentPlanner` song song với `MeetingSummarizer` trong pipeline).

### 7.2 `meetasr/realtime/worker.py` — luồng realtime (`doc_delta`)

Đây là chỗ hợp nhất với hạn chế #3 doc 13 (doc realtime hiện là text thô). Thiết kế:

```python
# Trong _run(), sau khi pipeline sẵn sàng:
planner_state = {"content_kind": None, "outline": None, "sections_written": set()}

# Sau khi tích luỹ đủ transcript của TOÀN BỘ audio đã xử lý tới thời điểm hiện tại
# (không phải chỉ 1 chunk 30s — outline cần nhìn đủ ngữ cảnh):
if planner_state["outline"] is None and accumulated_chars >= SAMPLE_CHARS_FOR_PLAN:
    content_kind, outline = planner.plan_only(accumulated_text)
    planner_state.update(content_kind=content_kind, outline=outline)
    for sec in outline:
        _emit(source_id, {"type": "doc_delta", "section_id": sec["id"],
                           "heading": sec["heading"], "markdown": "*Đang viết...*"})

# Mỗi khi có thêm text mới VÀ đã có outline: viết/refresh từng section liên quan
if planner_state["outline"]:
    for sec in planner_state["outline"]:
        doc_section = planner.write_one_section(sec, planner_state["content_kind"], new_text_since_last_write)
        if doc_section.markdown:
            _emit(source_id, {"type": "doc_delta", "section_id": sec["id"],
                               "heading": sec["heading"], "markdown": doc_section.markdown})
```

**Quan trọng — kiểm soát chi phí:** gọi LLM cho MỌI section mỗi lần có text mới là tốn kém. Áp
dụng ngưỡng: chỉ trigger viết lại khi tích luỹ đủ `MIN_CHARS_TRIGGER` (vd 1500 ký tự) văn bản mới
kể từ lần viết trước — đúng như đề xuất "1500-2500 ký tự" đã có ở doc 11 mục 3.2. Toàn bộ tính
năng realtime-doc nên **tắt mặc định**, bật qua config `realtime_doc: true` (đã ghi trong doc 13
mục 3) vì chi phí LLM liên tục là đáng kể.

---

## 8. Migration & tương thích ngược

| Việc | Có phá vỡ gì không? |
|---|---|
| Thêm `schemas_doc.py`, `planner.py`, `llm/prompts/plan_vi.txt`, `write_section_vi.txt` | Không — file mới hoàn toàn |
| Thêm `load_generic_prompts()` vào `prompts.py` | Không — hàm mới, không sửa `load_prompts()` cũ |
| Sửa `doc_gen.build_summary()` (Phase 2, chưa ai dùng ngoài nhóm) | Không ảnh hưởng Phase 1 |
| `MeetingSummarizer`, `MeetingReport`, route `/v1/meeting/summarize` (Phase 1) | **Giữ nguyên, không đụng** |
| 8 file prompt domain-cứng hiện có (`*_meeting_vi.txt`, `*_consultation_vi.txt`) | Giữ lại cho đến khi `DocumentPlanner` chạy ổn định trong Phase 2, sau đó archive (không xoá vội — Phase 1 route vẫn phụ thuộc `load_prompts()` cũ) |

**Cách nạp `DocumentPlanner` song song với `MeetingSummarizer` trong `AutoPipeline`:** thêm 1
attribute mới, không thay attribute cũ:

```python
# auto_pipeline.py — cạnh _build_llm() hiện có, thêm:
@staticmethod
def _build_doc_planner(llm_cfg: dict) -> Any:
    """Build a DocumentPlanner from the same LLM config block as _build_llm."""
    from meetasr.llm.planner import DocumentPlanner
    from meetasr.register import tables

    provider = llm_cfg.get("provider", "openai")
    client_kwargs = {k: v for k, v in llm_cfg.items()
                     if k not in ("provider", "language", "temperature", "max_tokens")}
    if "api_key" in client_kwargs and str(client_kwargs["api_key"]).startswith("${"):
        client_kwargs["api_key"] = os.environ.get(client_kwargs["api_key"][2:-1], "")
    client = tables.llm_classes.get(provider)(**client_kwargs)

    return DocumentPlanner(
        client=client,
        language=llm_cfg.get("language", "vi"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 4096),
    )
```

Rồi trong `MeetPipeline.__init__` gán `self.doc_planner = ...` bên cạnh `self.summarizer = ...`
đã có. `realtime.py`/`worker.py` đọc `pipeline.doc_planner`, route Phase 1 vẫn đọc
`pipeline.summarizer` như cũ — hai đường chạy song song, không giẫm chân nhau.

---

## 9. Frontend — thay đổi tối thiểu

`frontend-next/src/lib/types.ts` — thêm trường tuỳ chọn, không phá kiểu cũ:

```ts
export interface DocSection {
  id: string;
  heading: string;
  kind?: string;   // mới: "summary" | "action_items" | ... | bất kỳ chuỗi nào — không ràng buộc union
  markdown: string;
  writing?: boolean;
}
```

`ProcessingView.tsx` / `DocumentView.tsx` **không bắt buộc phải đổi** — chúng đã render
`heading` + `markdown` mà không quan tâm `kind`. Việc dùng `kind` để chọn icon/màu theo loại mục
(vd icon riêng cho "action_items" vs "quotes") là cải tiến UI tuỳ chọn, không phải điều kiện để
chạy được — có thể làm sau, không chặn AI Engineer 2.

---

## 10. Kế hoạch kiểm thử

1. **Unit test `_chunk()`:** input nhiều dòng dài hơn `MAX_CHARS_PER_CHUNK` → xác nhận không dòng
   nào bị cắt giữa chừng (so tổng số dòng input/output).
2. **Unit test `_representative_sample()`:** input ngắn hơn `max_chars` → trả nguyên văn; input
   dài hơn → độ dài kết quả ≈ `max_chars`, chứa cả phần đầu/giữa/cuối.
3. **Test outline đa dạng theo nội dung** (mock `client.chat` trả JSON cố định, không gọi LLM
   thật): 3 input khác domain → 3 outline có `kind` khác nhau, xác nhận không luôn ra đúng
   `["summary","topics","action_items","decisions"]` như bộ cũ.
4. **Test "không bịa nội dung":** mock `_call_write` trả `""` cho 1 section → xác nhận section đó
   bị loại khỏi kết quả cuối (không xuất hiện trong `DocumentReport.sections`).
5. **Test end-to-end (cần LLM thật/Ollama local):** 3 file audio khác loại (họp, bài giảng, hội
   thoại phiếm) → so sánh outline + đọc thử nội dung từng mục, xác nhận không mục nào bị ép vào
   khuôn "action_items/decisions" khi nội dung không có.
6. **Test chi phí:** đếm số lần gọi `client.chat` cho transcript 90 phút → xác nhận có nhiều lượt
   map-reduce (không phải 1 lượt duy nhất như hạn chế cũ của `_truncate` no-op).

---

## 11. Việc KHÔNG làm ở bản đầu (out of scope, ghi chú cho sau)

- Không cần streaming từng token của LLM ra frontend — chỉ cần streaming theo **section hoàn
  chỉnh** (đã đủ cho trải nghiệm "tài liệu hình thành dần").
- Không cần cache/join outline giữa các lần finalize khác nhau của cùng 1 source — mỗi lần
  finalize chạy planner mới (đơn giản, chi phí chấp nhận được ở quy mô hiện tại).
- Không cần chạy `_write_section` song song ở bản đầu (đơn giản trước, tối ưu tốc độ sau nếu đo
  thấy chậm — ghi chú trong code bằng comment, không làm sớm khi chưa có số liệu thật).
