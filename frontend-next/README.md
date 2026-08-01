# MeetingMind — Frontend (Phase 2)

Giao diện base cho sản phẩm Phase 2 ("NotebookLM cho audio/video"). Đây là **thiết kế nền**:
UI hoàn chỉnh chạy trên **mock data**, để 4 thành viên nối vào API FastAPI thật.

Kế hoạch chi tiết: [`../meet_docs/docs/12_frontend_nextjs_plan.md`](../meet_docs/docs/12_frontend_nextjs_plan.md).

## Chạy dev

```bash
corepack enable
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm dev            # http://localhost:3000
```

`next.config.ts` proxy các API nhỏ `/v1/*` → FastAPI. Upload media lớn và yêu
cầu finalize đi trực tiếp từ browser tới FastAPI. Finalize trả `202` ngay, sau
đó giao diện theo dõi job tạo tài liệu bằng WebSocket
`/v1/document-jobs/{id}/events`.

- `MEETASR_API`: URL backend cho Server Component/rewrite.
- `NEXT_PUBLIC_MEETASR_API`: URL backend mà browser dùng để upload/finalize.
- `NEXT_PUBLIC_WS_HOST`: host WebSocket Job/live mic.

Sao chép `.env.example` thành `.env.local` nếu backend không chạy ở host mặc
định. OAuth là tùy chọn vì giao diện hỗ trợ chế độ khách. Nếu bật đăng nhập,
điền `NEXTAUTH_URL`, `NEXTAUTH_SECRET` và credentials Google/GitHub; giá trị
`NEXTAUTH_SECRET` phải giống `JWT_SECRET` của backend trong flow local hiện tại.

## Ngôn ngữ thiết kế — "Transcription Studio"

Chủ đề sản phẩm là sự **chuyển hoá giọng nói → tài liệu có cấu trúc**. Mọi lựa chọn xoay quanh đó.
Toàn bộ token nằm trong [`src/app/globals.css`](src/app/globals.css) (`@theme`).

- **Màu (nền sáng):** giấy `#F5F6F8`, surface trắng, mực `#16161D`. Accent chính **ultramarine
  `#2C3EE0`**. Accent phụ **coral `#FF6B4A`** — chỉ dùng cho trạng thái "đang nghe / đang xử lý".
- **Chữ:** display **Bricolage Grotesque**, body **Inter**, mono **Space Mono** (timestamp,
  nhãn speaker, số liệu — vì transcript là "dữ liệu có mốc thời gian").
- **Signature:** màn hình xử lý 2 cột (lời thoại ↔ tài liệu hình thành dần) + motif **waveform**
  chuyển động khi đang nghe. Dùng class tiện ích: `.eyebrow`, `.caret`, `.font-display`,
  `.animate-rise`, và component `<Waveform live />`.

Chất lượng nền: responsive tới mobile, focus ring rõ, tôn trọng `prefers-reduced-motion`.

## Cấu trúc

```
src/
├── app/
│   ├── layout.tsx              # fonts + shell (Sidebar)
│   ├── page.tsx                # Thư viện (Library)
│   ├── upload/page.tsx         # Tải lên (dropzone)
│   └── sources/[id]/page.tsx   # branch: đang xử lý → ProcessingView | xong → DocumentView
├── components/
│   ├── Sidebar, SourceCard, StageProgress, FinalizeDialog
│   ├── ProcessingView.tsx      # ★ màn hình signature (đang mô phỏng streaming)
│   ├── DocumentView.tsx        # tabs Tài liệu / Lời thoại / Media + Xuất
│   ├── MarkdownLite.tsx        # tạm — thay bằng react-markdown khi nối backend
│   └── ui.tsx                  # PageHeader, StatusBadge, SpeakerChip, Waveform
└── lib/
    ├── types.ts                # ★ hợp đồng dữ liệu (khớp doc 11) — nối API theo đây
    ├── mock.ts                 # dữ liệu giả, thay bằng fetch
    └── format.ts               # format thời lượng, timestamp, màu speaker
```

## Việc tiếp theo của nhóm (thay mock → thật)

1. **Types là hợp đồng:** `src/lib/types.ts` khớp protocol trong doc 11. Sinh types thật từ
   OpenAPI của FastAPI (`openapi-typescript`) và thay dần.
2. **Data thật:** thay `MOCK_SOURCES` / `getSource` bằng React Query gọi `/v1/sources`.
3. **Realtime:** thay phần `useEffect` mô phỏng trong `ProcessingView` bằng hook `useJobEvents`
   (WebSocket `/v1/jobs/{id}/events`) — thiết kế ở doc 12 §4.
4. **Finalize / Export:** nối `FinalizeDialog` và menu Xuất trong `DocumentView` vào endpoint thật.
