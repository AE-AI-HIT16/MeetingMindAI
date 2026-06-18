from meetasr.llm.ollama_client import OllamaClient
from meetasr.utils.io import read_txt


def test_ollama_basic_chat():
    client = OllamaClient(model="llama3")
    path_promt = "meetasr/llm/prompts/decisions_vi.txt"
    path_transcript = "tests/test_llm_transcrip_example/test_01.txt"

    prompt = read_txt(path_promt)
    transcipt = read_txt(path_transcript)

    prompt = prompt.replace("{transcript}", transcipt)

    response = client.chat(prompt=prompt)

    print(response)

    assert isinstance(response, str)
    assert len(response) > 0