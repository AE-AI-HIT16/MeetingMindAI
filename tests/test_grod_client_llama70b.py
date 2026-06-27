from meetasr.llm.groq_client import GroqClient
from meetasr.utils.io import read_txt
from dotenv import load_dotenv
from meetasr.llm.llm_utils.convert_format_input import format_sentence_info
import os
import json
import pprint

load_dotenv()


def test_groq_basic_chat():
    client = GroqClient(
        api_key=os.getenv("GROD_API_KEY_KHANH"),
        model="llama-3.3-70b-versatile"
    )

    path_prompt = "meetasr/llm/prompts/decisions/decisions_consultation_vi.txt"
    path_transcript = "tests/data/consultation_mock_v2.json"

    prompt = read_txt(path_prompt)

    with open(path_transcript, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentence_info = data["sentence_info"]


    transcript = format_sentence_info(sentence_info)

    prompt = prompt.replace("{transcript}", transcript)

    # pprint.pprint(promt)

    response = client.chat(prompt=prompt)

    pprint.pprint(response)

    assert isinstance(response, str)
    assert len(response) > 0