from meetasr.streaming.processor.processor import AudioProcessor
from meetasr.streaming.model_manager import ModelManager

# Không có tác dụng gì, chỉ để ví dụ về 1 processer
class DummyProcessor(AudioProcessor):

    def __init__(self, model_manager: ModelManager):
        self.model = model_manager.get_model("Dummy")

    def process(self, audio: bytes):

        return {
            "bytes": len(audio)
        }