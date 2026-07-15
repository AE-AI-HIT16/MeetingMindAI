import asyncio


# Nơi khởi chạy luồng logic
class AudioWorker:

    def __init__(self, session, processor):

        self.session = session

        # Cần đổi sang list processer để chứa nhiều processer
        self.processor = processor

    async def run(self):

        loop = asyncio.get_running_loop()

        while True:

            # Lấy dữ liệu từ trong queue
            audio = await self.session.audio_queue.get()

            # Đưa gữ liệu cho processer xử lý và nhận kết quả
            # (chỗ này cần gọi nhiều và thực hiện tuần tự nếu xử lý cần nhiều bước logic
            # Gọi các model vào xử lý ở đây
            # process có kiểu dữ liệu là AudioProcessor
            result = await loop.run_in_executor(
                None,
                self.processor.process,
                audio,
            )

            # Gửi kết quả cho websocket
            if result is not None:
                await self.session.websocket.send_json(result)

            self.session.audio_queue.task_done()