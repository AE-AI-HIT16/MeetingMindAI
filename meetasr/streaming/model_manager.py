from meetasr.streaming.init_model.asr import ZipformerViLoader
from meetasr.streaming.init_model.cam_plus import CAMPlusPlusLoader
from meetasr.streaming.init_model.punctuation import ViBERTCaPuLoader
from meetasr.streaming.init_model.vad import FsmnVADLoader


# Quản lý các model
class ModelManager:

    def __init__(self):
        self.models = {}
        self.locks = {}


    async def load_all(self):

        # Các model này chỉ là ví dụ, cần đổi sang model thật khi dùng
        self.models["vad"] = FsmnVADLoader.load()

        self.models["asr"] = ZipformerViLoader()

        self.models["cam_plus"] = CAMPlusPlusLoader()

        self.models["punctuation"] = ViBERTCaPuLoader()


    def get_model(self, name):
        return self.models[name]

    def get_lock(self, name):
        return self.locks[name]