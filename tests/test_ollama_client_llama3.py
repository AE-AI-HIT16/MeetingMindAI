import json
import os

import pytest

from meetasr.llm.llm_utils.chunking import run_pipeline_with_markdown
from meetasr.llm.llm_utils.validate import process_transcript
from meetasr.llm.ollama_client import OllamaClient
from meetasr.utils.io import read_txt

with open("tests/data/meeting_mock.json", "r", encoding="utf-8") as f:
    mock_meeting = json.load(f)

@pytest.mark.skipif(
    os.environ.get("MEETASR_RUN_LIVE_TESTS") != "1",
    reason="requires a running Ollama server; set MEETASR_RUN_LIVE_TESTS=1",
)
def test_ollama_basic_chat():
    client = OllamaClient(model="llama3")
    path_promt = ("meetasr/llm/prompts/decisions/decisions_meeting_vi.txt")

    prompt = read_txt(path_promt)
    # transcipt = read_txt(path_transcript)

    # Load data
    sentence_info = mock_meeting["sentence_info"]

    # Chunking
    chunks, markdown = run_pipeline_with_markdown(
        sentence_info=sentence_info,
        window_size=3,
        threshold=0.75,
        use_adaptive_threshold=False
    )

    # Validate
    process_transcript(mock_meeting)

    prompt = prompt.replace("{transcript}", markdown)

    response = client.chat(prompt=prompt)

    print(response)

    assert isinstance(response, str)
    assert len(response) > 0
