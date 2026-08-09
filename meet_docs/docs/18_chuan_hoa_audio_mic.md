# Chuẩn hoá audio từ mic → mono 16kHz PCM16 (giải pháp hiện tại)

> Trả lời trực tiếp câu hỏi: "không biết kiểu dữ liệu từ mic là gì để chuẩn hoá".
> Giải pháp đã có sẵn trong `frontend-next/src/hooks/useLiveMic.ts` — tài liệu này
> giải thích nguyên lý để cả team hiểu và không phải đoán bằng log.

## 1. Điểm mấu chốt: KHÔNG cần đoán định dạng mic

Đây là hiểu lầm phổ biến. Trong **trình duyệt**, bạn KHÔNG nhận raw bytes từ driver mic
(48000Hz/44100Hz, 16-bit/24-bit, mono/stereo… mỗi máy một kiểu). Web Audio API **đã chuẩn
hoá sẵn** cho bạn:

- Mọi mẫu âm thanh đến tay bạn luôn là **`Float32`, giá trị trong khoảng [-1.0, 1.0]**.
- Không phải int16, không phải byte, không phải 24-bit. Luôn luôn Float32 chuẩn hoá.
- Trình duyệt tự lo phần "đọc từ driver và đổi sang Float32".

Nên câu hỏi "nhận vào là loại nào?" có câu trả lời cố định: **Float32 [-1,1]**. Không cần
`console.log` để dò từng máy.

Thứ **duy nhất** thay đổi giữa các máy là **sample rate** — và bạn đọc được nó bằng 1 dòng,
không phải đoán (xem mục 3).

## 2. Luồng chuẩn hoá (3 bước)

```
Mic (driver: 48kHz? 44.1kHz? — trình duyệt tự lo)
  │
  ▼  Web Audio API chuẩn hoá
Float32 [-1,1] @ ctx.sampleRate  (thường 48000)   ← AudioWorklet gửi ra ở đây
  │
  ├─ Bước A: gộp về mono (nếu cần) — ta ép mono ngay từ getUserMedia
  ├─ Bước B: hạ tần số 48kHz → 16kHz  (downsampleTo16k)
  └─ Bước C: Float32 → PCM16 little-endian  (floatToPCM16)
  │
  ▼
Bytes PCM16 mono 16kHz  →  gửi qua WebSocket  →  backend
```

Backend `/v1/live` mong đợi đúng: **PCM16 little-endian, mono, 16kHz, frame ~250ms**
(xem [meetasr/api/routes/live.py](../../meetasr/api/routes/live.py)).

## 3. Cách "xem thông số" mà không cần log mò

Ba thông số cần biết, lấy được ngay bằng code (không phải đoán):

```ts
const stream = await navigator.mediaDevices.getUserMedia({
  audio: { channelCount: 1 },   // ← ÉP mono ngay tại đây, hết lo stereo
});

// 1. Sample rate thực tế của mic:
const ctx = new AudioContext();
console.log("sample rate =", ctx.sampleRate);   // vd 48000

// 2. Xem track mic báo cáo gì (số kênh, thiết bị…):
const track = stream.getAudioTracks()[0];
console.log(track.getSettings());
// { sampleRate: 48000, channelCount: 1, echoCancellation: true, ... }

// 3. Biên độ: trong Web Audio LUÔN là Float32 [-1,1]. Không cần đo.
```

`track.getSettings()` là "công cụ xem thông số" mà bạn đang tìm — nó trả về đúng những gì
mic đang dùng, thay cho việc log mò từng byte.

## 4. Code chuẩn hoá thực tế (đã có trong dự án)

Toàn bộ nằm ở [frontend-next/src/hooks/useLiveMic.ts](../../frontend-next/src/hooks/useLiveMic.ts).
Ba mảnh chính:

**Bước A — ép mono ngay khi xin quyền mic** (không phải xử lý stereo về sau):
```ts
await navigator.mediaDevices.getUserMedia({
  audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
});
```

**Bước B — hạ 48kHz → 16kHz** (`downsampleTo16k`, dòng 22-37):
```ts
// ratio = 48000/16000 = 3  →  gộp trung bình mỗi 3 mẫu thành 1 (chống aliasing nhẹ)
const ratio = inRate / 16000;
// ... lấy trung bình cửa sổ nhỏ, KHÔNG cần thư viện
```
Lý do lấy trung bình thay vì "bỏ bớt mẫu": bỏ mẫu thô gây méo tần số cao (aliasing);
trung bình cửa sổ nhỏ là bộ lọc rẻ, đủ tốt cho tiếng nói.

**Bước C — Float32 → PCM16 little-endian** (`floatToPCM16`, dòng 39-48):
```ts
const s = Math.max(-1, Math.min(1, sample));      // kẹp về [-1,1] cho an toàn
view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);  // true = little-endian
```
- `* 0x8000` (32768) cho giá trị âm, `* 0x7fff` (32767) cho dương — đúng dải int16.
- `true` cuối = little-endian (backend `np.int16` đọc little-endian).

**Gom frame ~250ms rồi mới gửi** (dòng ~120): 16000 × 0.25 = **4000 mẫu/frame**. Gửi
từng chút một thì tốn round-trip; gom 250ms là mức backend mong đợi.

## 5. Sai lầm thường gặp (để khỏi mất 9 phần log)

| Sai lầm | Hậu quả | Đúng ra |
|---|---|---|
| Cố đọc raw bytes/bit-depth từ mic | Không đọc được — trình duyệt che đi | Nhận Float32 [-1,1], khỏi quan tâm bit-depth |
| Hardcode 48000 | Máy 44100Hz ra giọng méo/nhanh-chậm | Đọc `ctx.sampleRate` động |
| Downsample bằng cách bỏ mẫu | Méo tần số cao | Trung bình cửa sổ (`downsampleTo16k`) |
| Quên `channelCount: 1` | Nhận stereo, gửi sai/đôi dữ liệu | Ép mono tại `getUserMedia` |
| Gửi big-endian | Backend đọc ra nhiễu trắng | `setInt16(..., true)` little-endian |
| Gửi từng mẫu / block nhỏ | Nhiều round-trip, trễ | Gom 4000 mẫu (~250ms) rồi gửi |

## 6. Cách tự kiểm tra đã chuẩn hoá đúng chưa

Không cần backend. Ghi 2-3s rồi phát lại chính bytes PCM16 bạn tạo ra:
- Nếu nghe **đúng giọng, đúng tốc độ** → chuẩn hoá đúng.
- Nếu **nhanh/chậm bất thường** → sai sample rate (bước B).
- Nếu **nhiễu trắng/rè** → sai endian hoặc sai scale int16 (bước C).
- Nếu **méo the thé** → downsample kiểu bỏ mẫu (dùng trung bình thay thế).

Hoặc dùng test client backend đã có (phát WAV Việt qua WS `/v1/live`) để so text — nếu ra
chữ đúng tiếng Việt thì cả chuỗi chuẩn hoá + protocol đã đúng.

## Tóm tắt cho bạn đang mắc

1. **Đừng đoán định dạng mic** — trình duyệt luôn đưa bạn Float32 [-1,1]. Chỉ sample rate
   là biến, đọc bằng `ctx.sampleRate`.
2. Ép mono ngay tại `getUserMedia({audio:{channelCount:1}})`.
3. Ba hàm chuẩn hoá đã viết sẵn trong `useLiveMic.ts` — đọc `downsampleTo16k` và
   `floatToPCM16`, không cần viết lại.
4. Kiểm tra bằng cách phát lại PCM16 tự tạo (mục 6), không cần log từng byte.
