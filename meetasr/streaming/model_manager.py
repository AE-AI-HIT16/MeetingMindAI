from meetasr.streaming.processor.dummy_processor import DummyProcessor


# Quản lý các model
class ModelManager:

    def __init__(self):
        self.models = {}
        self.locks = {}


    async def load_all(self):

        # Các model này chỉ là ví dụ, cần đổi sang model thật khi dùng
        self.models["asr"] = WhisperModel(
            "large-v3",
            device="cuda"
        )

        self.models["vad"] = VADModel(
            "silero",
            device="cuda"
        )

        self.models["diarizer"] = DiarizerModel(
            device="cuda"
        )

        self.models["DummyProcessor"] = Dummy(
            device="cuda"
        )


    def get_model(self, name):
        return self.models[name]

    def get_lock(self, name):
        return self.locks[name]