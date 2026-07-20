from modelscope.hub.snapshot_download import snapshot_download

from meetasr.models.spk.campplus import CAMPlusPlus


class CAMPlusPlusLoader:
    """Utility class for downloading and loading the CAM++ speaker model."""

    MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
    REVISION = "v1.0.0"

    @classmethod
    def load(
        cls,
        device: str = "cpu",
        revision: str |None = None,
        **kwargs,
    ) -> CAMPlusPlus:
        """Download (if needed) and load the CAM++ model."""

        model_path = snapshot_download(
            model_id=cls.MODEL_ID,
            revision=revision or cls.REVISION,
        )

        return CAMPlusPlus(
            model_path=model_path,
            device=device,
            **kwargs,
        )