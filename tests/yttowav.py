import yt_dlp
from yt_dlp.utils import download_range_func # Bắt buộc import thêm hàm này

# Cấu hình tải và cắt
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav'
    }],
    # Truyền khoảng thời gian (giây) vào mảng: (start_time, end_time)
    'download_ranges': download_range_func(None, [(2593, 3369)]),
    # Bạn có thể đổi tên file xuất ra cho gọn gàng (tuỳ chọn)
    'outtmpl': 'audio_cut.%(ext)s'
}

URL = "https://www.youtube.com/watch?v=91LXa-bwyIM"

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL])