from modelscope.hub.snapshot_download import snapshot_download

from meetasr.models.vad.fsmn_vad import FsmnVAD


class FsmnVADLoader:
    """Utility class for downloading and loading the FSMN-VAD model."""

    MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    REVISION = "v2.0.4"

    @classmethod
    def load(
        cls,
        device: str = "cpu",
        revision: str | None = None,
        **kwargs,
    ) -> FsmnVAD:
        """Download (if needed) and load the FSMN-VAD model.

        Args:
            device: Inference device ("cpu", "cuda", ...).
            revision: Optional ModelScope model revision.
            **kwargs: Additional arguments passed to FsmnVAD.

        Returns:
            Initialized FsmnVAD instance.
        """
        model_path = snapshot_download(
            model_id=cls.MODEL_ID,
            revision=revision or cls.REVISION,
        )

        return FsmnVAD(
            model_path=model_path,
            device=device,
            **kwargs,
        )