"""Check RunPod cold start + Network Volume model cache.

Sends one "warmup" job (the same one the backend sends when a realtime session
opens), waits for it and prints where models were loaded from.

Usage (on EC2, where the backend .env lives):
    set -a; . ~/Backend/.env; set +a; python3 check_runpod_volume.py

Cost: ~1 s GPU if a worker is already up; one model load (tens of seconds) on
a cold worker.
"""

import json
import os
import time
import urllib.request

BASE = os.environ.get("RUNPOD_API_BASE_URL", "https://api.runpod.ai/v2").rstrip("/")
URL = f"{BASE}/{os.environ['RUNPOD_ENDPOINT_ID']}"
HEADERS = {
    "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
    "Content-Type": "application/json",
}


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def gb(n):
    return f"{(n or 0) / 1e9:.2f} GB"


health = call("/health")
print("Workers trước khi gửi job:", health["workers"])

t0 = time.monotonic()
job = call("/run", {"input": {"action": "warmup"}})
print("Đã gửi job warmup:", job["id"])
while True:
    status = call(f"/status/{job['id']}")
    if status["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        break
    print(f"  {time.monotonic() - t0:5.0f}s  {status['status']}")
    time.sleep(3)

total = time.monotonic() - t0
out = status.get("output") or {}
print()
print(f"Tổng thời gian đến khi sẵn sàng: {total:.1f}s")
print(f"  delayTime (xếp hàng + boot container): {status.get('delayTime', 0) / 1000:.1f}s")
print(f"  executionTime (gồm load model nếu cold): {status.get('executionTime', 0) / 1000:.1f}s")
print("Output:", json.dumps(out, indent=2, ensure_ascii=False))
print()

if status["status"] != "COMPLETED":
    print("❌ Job lỗi:", status.get("error"))
elif "network_volume_mounted" not in out:
    print("⚠️  Worker đang chạy image cũ (chưa có chẩn đoán). Chờ CI build image mới + endpoint trỏ đúng image rồi chạy lại.")
elif not out["network_volume_mounted"]:
    print("❌ KHÔNG thấy /runpod-volume → Network Volume chưa gắn vào endpoint, hoặc volume khác data center với GPU của endpoint.")
elif "model_load_seconds" not in out:
    print("ℹ️  Volume ĐÃ gắn, nhưng worker đã warm sẵn (không cold start) → chưa đo được tốc độ load.")
    print("   Muốn đo cold start: Max Workers = 0 → Save → đặt lại như cũ → Save, rồi chạy lại script.")
elif out.get("hf_cache_bytes_before_load", 0) > 0:
    print(f"✅ Volume HOẠT ĐỘNG: cache đã có {gb(out['hf_cache_bytes_before_load'])} trước khi load → model đọc từ volume, "
          f"load mất {out['model_load_seconds']}s.")
else:
    print(f"🟡 Volume đã gắn nhưng cache trống trước khi load → lần này model được TẢI về và lưu vào volume "
          f"({gb(out.get('hf_cache_bytes'))}, {out['model_load_seconds']}s). Ép cold start lần nữa rồi chạy lại: "
          "lần sau phải ra ✅ và load nhanh hơn.")
