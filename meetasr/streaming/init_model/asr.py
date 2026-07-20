from huggingface_hub import snapshot_download

from meetasr.models.asr.zipformer_vi import ZipformerViASR


class ZipformerViLoader:
    REPO_ID = "hynt/Zipformer-30M-RNNT-6000h"

    @classmethod
    def load(
        cls,
        model_dir: str = "./models/zipformer-30m-rnnt-6000h",
        device: str = "cpu",
    ) -> ZipformerViASR:
        """Download (if needed) and load ZipformerViASR."""

        model_path = snapshot_download(
            repo_id=cls.REPO_ID,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )

        return ZipformerViASR(
            model_path=model_path,
            device=device,
        )