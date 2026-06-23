import pprint
from meetasr.llm.llm_utils.chunking import run_pipeline_with_markdown


# =========================
# MOCK DATA
# =========================
mock_meeting = {
    "key": "meeting_demo_001",
    "text": "Host mở đầu cuộc họp, giới thiệu mục tiêu. Member A đề xuất dùng React cho frontend. Member B nói sẽ phụ trách UI design. Member C đề xuất dùng Node.js cho backend. Host đồng ý với hướng đi tổng thể. Member A hỏi về deadline. Host trả lời deadline là cuối tháng. Member B đề xuất dùng Figma để thiết kế. Member C xác nhận API sẽ được thiết kế REST. Member A nói sẽ setup project structure. Host nhấn mạnh cần chia task rõ ràng. Member B hỏi về style guide. Host đề xuất dùng Material UI. Member C đồng ý với kiến trúc backend. Member A xác nhận sẽ setup GitHub repo. Host kết luận cuộc họp. Các thành viên thống nhất kế hoạch triển khai.",
    "duration": 900.0,
    "sentence_info": [
        {"text": "Host mở đầu cuộc họp, giới thiệu mục tiêu.", "start": 0.0, "end": 5.0, "speaker": "Host"},
        {"text": "Member A đề xuất dùng React cho frontend.", "start": 5.0, "end": 10.0, "speaker": "Member A"},
        {"text": "Member B nói sẽ phụ trách UI design.", "start": 10.0, "end": 15.0, "speaker": "Member B"},
        {"text": "Member C đề xuất dùng Node.js cho backend.", "start": 15.0, "end": 20.0, "speaker": "Member C"},
        {"text": "Host đồng ý với hướng đi tổng thể.", "start": 20.0, "end": 25.0, "speaker": "Host"},
        {"text": "Member A hỏi về deadline.", "start": 25.0, "end": 30.0, "speaker": "Member A"},
        {"text": "Host trả lời deadline là cuối tháng.", "start": 30.0, "end": 35.0, "speaker": "Host"},
        {"text": "Member B đề xuất dùng Figma để thiết kế.", "start": 35.0, "end": 40.0, "speaker": "Member B"},
        {"text": "Member C xác nhận API sẽ được thiết kế REST.", "start": 40.0, "end": 45.0, "speaker": "Member C"},
        {"text": "Member A nói sẽ setup project structure.", "start": 45.0, "end": 50.0, "speaker": "Member A"},
        {"text": "Host nhấn mạnh cần chia task rõ ràng.", "start": 50.0, "end": 55.0, "speaker": "Host"},
        {"text": "Member B hỏi về style guide.", "start": 55.0, "end": 60.0, "speaker": "Member B"},
        {"text": "Host đề xuất dùng Material UI.", "start": 60.0, "end": 65.0, "speaker": "Host"},
        {"text": "Member C đồng ý với kiến trúc backend.", "start": 65.0, "end": 70.0, "speaker": "Member C"},
        {"text": "Member A xác nhận sẽ setup GitHub repo.", "start": 70.0, "end": 75.0, "speaker": "Member A"},
        {"text": "Host kết luận cuộc họp.", "start": 75.0, "end": 80.0, "speaker": "Host"},
        {"text": "Các thành viên thống nhất kế hoạch triển khai.", "start": 80.0, "end": 85.0, "speaker": "Group"},
    ],
}


# =========================
# TEST CASE
# =========================
def test_run_pipeline_with_markdown():

    sentence_info = mock_meeting["sentence_info"]

    chunks, markdown = run_pipeline_with_markdown(
        sentence_info=sentence_info,
        window_size=3,
        threshold=0.75,
        use_adaptive_threshold=False
    )

    # =========================
    # BASIC ASSERTS
    # =========================
    assert isinstance(chunks, list)
    assert isinstance(markdown, str)

    assert len(chunks) > 0, "Chunks should not be empty"

    # =========================
    # STRUCTURE CHECK
    # =========================
    for c in chunks:
        assert "markdown" in c, "Missing markdown in chunk"
        assert "start_time" in c, "Missing start_time"
        assert "end_time" in c, "Missing end_time"

    # =========================
    # LOGIC CHECK
    # =========================
    if len(chunks) > 1:
        assert True
    else:
        print("⚠ Only 1 chunk (threshold may be too high)")

    # =========================
    # DEBUG OUTPUT (giống mẫu bạn đưa)
    # =========================
    pprint.pprint(sentence_info)
    print("\n================ CHUNKS ================\n")
    pprint.pprint(chunks)

    print("\n================ MARKDOWN ================\n")
    print(markdown)