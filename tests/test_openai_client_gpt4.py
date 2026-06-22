from meetasr.llm.openai_client import OpenAIClient
from meetasr.utils.io import read_txt
from dotenv import load_dotenv
import os

load_dotenv()

def test_openai_basic_chat():
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),  # hoặc load từ env
        model="gpt-4o-mini"
    )

    path_prompt = "meetasr/llm/prompts/decisions_vi.txt"
    path_transcript = "tests/test_llm_transcrip_example/test_01.txt"

    prompt = read_txt(path_prompt)
    transcript = read_txt(path_transcript)

    prompt = prompt.replace("{transcript}", transcript)

    response = client.chat(prompt=prompt)

    print(response)

    assert isinstance(response, str)
    assert len(response) > 0