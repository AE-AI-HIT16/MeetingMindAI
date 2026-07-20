from huggingface_hub import snapshot_download

from meetasr.models.punc.vibert_capu import ViBERTCaPuPunc


class ViBERTCaPuLoader:
    REPO_ID = "dragonSwing/vibert-capu"

    @classmethod
    def load(
        cls,
        model_dir="./models/vibert-capu",
        device="cpu",
        **kwargs,
    ):
        model_path = snapshot_download(
            repo_id=cls.REPO_ID,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )

        return ViBERTCaPuPunc(
            model_path=model_path,
            device=device,
            **kwargs,
        )