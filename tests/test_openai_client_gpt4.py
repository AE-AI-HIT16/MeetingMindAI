from unittest.mock import patch

from meetasr.llm.openai_client import OpenAIClient
from meetasr.utils.io import read_txt
from types import SimpleNamespace


@patch.object(OpenAIClient, "chat")
def test_openai_basic_chat(mock_chat):
    mock_chat.return_value = """
## Quyết định
- Đồng ý triển khai.

## Công việc
- A phụ trách backend.
"""

    path_prompt = "meetasr/llm/prompts/decisions_meeting_vi.txt"
    path_transcript = "tests/test_llm_transcrip_example/test_01.txt"

    prompt = read_txt(path_prompt)
    transcript = read_txt(path_transcript)

    prompt = prompt.replace("{transcript}", transcript)

    client = OpenAIClient(
        api_key="fake-key",
        model="gpt-4o-mini",
    )

    response = client.chat(prompt=prompt)

    mock_chat.assert_called_once()

    assert isinstance(response, str)
    assert "backend" in response