# Phase 2 — Frontend Plan: Next.js + TypeScript

> Kế hoạch làm lại toàn bộ giao diện (thay các trang HTML tĩnh trong `frontend/`)
> bằng **Next.js + TypeScript**, phục vụ sản phẩm Phase 2
> ([11_phase2_notebooklm_plan.md](11_phase2_notebooklm_plan.md)): upload video/audio →
> docs hiện realtime → chọn tóm tắt/toàn văn → xuất PDF → thư viện lưu trữ.

---

## 1. Tech stack

| Lớp | Chọn | Lý do |
|---|---|---|
| Framework | **Next.js 15 (App Router) + TypeScript** | Chuẩn hiện tại; routing/file-based dễ học |
| UI | **Tailwind CSS + shadcn/ui** | Component đẹp sẵn, copy-vào-dự-án nên dễ tùy biến; không lock-in |
| Data fetching | **TanStack Query (React Query)** | Cache, refetch, trạng thái loading/error chuẩn hoá |
| Realtime | **WebSocket native** + custom hook `useJobEvents` | Chỉ 1 kênh sự kiện, không cần socket.io |
| State cục bộ | **Zustand** (chỉ khi cần) | Nhẹ; phần lớn state đã nằm trong React Query + URL |
| API types | **openapi-typescript** sinh types từ FastAPI `/openapi.json` | FE–BE không lệch schema; đây là lý do AI Eng 3 phải giữ OpenAPI chuẩn |
| Markdown render | **react-markdown + remark-gfm** | Docs là markdown; render section streaming |
| Media player | HTML5 `<video>/<audio>` + custom controls | Backend đã hỗ trợ HTTP Range |
| Deploy dev | `next dev` (port 3000) proxy tới FastAPI (port 8000) qua `next.config.ts rewrites` | Tránh CORS khi dev |

**Không dùng** (tránh over-engineering): Redux, socket.io, SSR data-fetching phức tạp
(app này gần như thuần client-side sau khi đăng nhập), monorepo tooling.

## 2. Cấu trúc thư mục

```
frontend-next/                     # dự án mới, giữ frontend/ cũ đến khi thay xong
├── src/
│   ├── app/
│   │   ├── layout.tsx             # shell: sidebar + theme
│   │   ├── page.tsx               # Library (trang chủ)
│   │   ├── upload/page.tsx        # Upload media
│   │   ├── sources/[id]/
│   │   │   ├── page.tsx           # Processing view (docs realtime) / Document viewer
│   │   │   └── loading.tsx
│   │   └── api-proxy/…            # (không cần nếu dùng rewrites)
│   ├── components/
│   │   ├── ui/                    # shadcn components
│   │   ├── library/SourceCard.tsx, SourceGrid.tsx
│   │   ├── upload/UploadDropzone.tsx, UploadProgress.tsx
│   │   ├── processing/LiveTranscript.tsx, LiveDocPane.tsx, StageProgress.tsx
│   │   ├── document/DocViewer.tsx, FinalizeChoiceDialog.tsx, ExportMenu.tsx
│   │   └── media/MediaPlayer.tsx
│   ├── hooks/
│   │   ├── useJobEvents.ts        # WebSocket → state (transcript, sections, status)
│   │   ├── useUpload.ts           # multipart upload + progress
│   │   └── useSources.ts, useDocument.ts   # React Query wrappers
│   ├── lib/
│   │   ├── api.ts                 # fetch client typed
│   │   ├── api-types.ts           # sinh bởi openapi-typescript (không sửa tay)
│   │   └── ws.ts                  # reconnect logic
│   └── styles/globals.css
├── next.config.ts                 # rewrites /v1/* → http://localhost:8000/v1/*
├── tailwind.config.ts
└── package.json
```

## 3. Các trang & luồng UX

### 3.1 Library — `/` (trang chủ)
- Grid card các source đã upload: thumbnail (icon audio/frame video), tên, thời lượng,
  ngày, badge trạng thái (`processing` / `done` / `failed`), badge loại docs đã có (summary/full).
- Click card → `/sources/[id]`. Nút Upload nổi bật. Empty state hướng dẫn khi chưa có gì.
- Xóa source (confirm dialog).

### 3.2 Upload — `/upload`
- Dropzone kéo-thả + chọn file; validate định dạng/dung lượng phía client trước khi gửi.
- Progress bar upload (dùng `XMLHttpRequest` hoặc `fetch` + stream để có % — axios cũng được).
- Upload xong → backend tạo job → **tự chuyển hướng** sang `/sources/[id]` (processing view).

### 3.3 Processing view — `/sources/[id]` khi job đang chạy (màn hình quan trọng nhất)
Layout 2 cột:
- **Trái — Live Transcript**: segment hiện dần (timestamp + speaker màu riêng + text),
  auto-scroll (tắt khi user cuộn tay).
- **Phải — Live Docs**: các section markdown hình thành dần; section đang được cập nhật
  có hiệu ứng nhấp nháy nhẹ. Đây chính là trải nghiệm "NotebookLM".
- Trên cùng: `StageProgress` (extracting → transcribing → generating) + % + thời gian.
- Nhận sự kiện qua `useJobEvents(jobId)`; reconnect tự động, replay đủ sự kiện cũ
  (backend hỗ trợ — xem doc 11 mục AI Eng 3).
- Khi `done` → mở **FinalizeChoiceDialog**: "📝 Tóm tắt" / "📄 Xuất toàn văn"
  (giải thích ngắn từng lựa chọn) → gọi `POST /v1/documents/{id}/finalize`.

### 3.4 Document viewer — `/sources/[id]` khi đã có docs
- Tab: **Docs** (markdown render đẹp) · **Transcript** (đầy đủ, tìm kiếm) · **Media**
  (player phát lại; nice-to-have: click segment transcript → tua media tới timestamp).
- `ExportMenu`: PDF / DOCX / Markdown (gọi endpoint export, tải file).
- Nút chạy lại finalize với mode kia (đã có full_text vẫn tạo thêm summary được).

## 4. Hook realtime — thiết kế `useJobEvents`

```ts
type JobState = {
  status: "connecting" | "processing" | "done" | "error";
  stage: string; progress: number;
  segments: TranscriptSegment[];       // append-only
  sections: Map<string, DocSection>;   // doc_delta ghi đè theo section_id
  error?: string;
};

function useJobEvents(jobId: string): JobState
// - Mở WS /v1/jobs/{jobId}/events, parse sự kiện theo protocol doc 11 mục 3.1
// - transcript_delta → push segments; doc_delta → set section theo id
// - onclose bất thường → reconnect (backoff 1s/2s/4s, tối đa 5 lần), server replay lại từ đầu
//   → phía client reset state khi reconnect để tránh trùng lặp
// - Job đã done từ trước (vào lại trang) → không mở WS, fetch REST 1 lần
```

Đây là phần logic khó nhất của FE — làm sớm (F3) với backend giả (mock WS server bằng
script Python phát lại sự kiện từ file JSONL) để không phải chờ backend thật.

## 5. Milestones frontend

| MS | Nội dung | Verify |
|---|---|---|
| F0 | Setup: Next.js + TS + Tailwind + shadcn, layout shell, proxy `/v1` về FastAPI, sinh `api-types.ts` từ OpenAPI | `pnpm dev` chạy; gọi `GET /v1/meetings` (Phase 1) hiện data thật |
| F1 | Design system: theme màu, typography, dark mode, các component nền (Card, Dialog, Badge) | Storybook-style trang `/dev/components` xem đủ component |
| F2 | Library + Upload hoàn chỉnh (với API thật của Data Eng — M0) | Upload mp4 → card xuất hiện trong Library |
| F3 | Processing view + `useJobEvents` (chạy với **mock WS** trước, backend thật sau) | Transcript + docs hiện dần mượt; reconnect giữa chừng không mất dữ liệu |
| F4 | FinalizeChoiceDialog + Document viewer + Export + Media player | Chọn tóm tắt → xem docs → tải PDF mở được |
| F5 | Polish: responsive, loading/error states, empty states, xóa source | Duyệt toàn bộ flow trên mobile viewport không vỡ |

Mapping với backend: F2 cần backend M0; F3 cần M2+M3 (trước đó dùng mock); F4 cần M4+M5.
Tuần 5 (M6 chung) là điểm hẹn tích hợp toàn bộ.

## 6. Kiến thức cần học (cho người làm FE)

1. **Next.js App Router** — official tutorial: https://nextjs.org/learn (phần Dashboard App, ~1 ngày).
2. **TypeScript cho React** — https://react-typescript-cheatsheet.netlify.app/ (đọc phần cơ bản).
3. **TanStack Query** — https://tanstack.com/query/latest/docs/framework/react/overview
   (concepts: queries, mutations, invalidation).
4. **shadcn/ui + Tailwind** — https://ui.shadcn.com/docs (cách add component, theming).
5. **WebSocket trong React** — pattern quản lý WS trong `useEffect` + cleanup; hiểu vì sao
   cần reconnect + replay (bài tập: viết hook nhận tick từ 1 WS echo server).
6. **openapi-typescript** — https://openapi-ts.dev/ (sinh types, dùng với `fetch`).

Bài tập khởi động (2–3 ngày): dựng Next.js app nhỏ có 1 trang list gọi FastAPI Phase 1
(`GET /v1/meetings`) qua React Query + types sinh từ OpenAPI, có dark mode. Xong bài này là xong F0.
