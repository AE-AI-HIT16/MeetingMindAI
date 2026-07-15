from abc import ABC, abstractmethod


# Lớp trừ tượng, cho phép các thao tác xử lý kế thừa nó và trở thành lớp
class AudioProcessor(ABC):

    @abstractmethod
    def process(self, audio: bytes):
        """
        Hàm xử lý audio.

        Người làm AI sẽ implement hàm này.
        """
        raise NotImplementedError