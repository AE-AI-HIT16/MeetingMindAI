// Mock data — replaced by real API calls once the backend lands.
// Kept in one place so the team can see the exact shapes the UI expects.

import type { DocSection, Source, TranscriptSegment } from "./types";

export const MOCK_SOURCES: Source[] = [
  {
    id: "live-1",
    title: "Họp kế hoạch Sprint 12",
    mediaType: "video",
    durationMs: 41 * 60 * 1000,
    createdAt: "2026-07-13T09:12:00",
    status: "processing",
    docs: ["live"],
  },
  {
    id: "s-2",
    title: "Phỏng vấn khách hàng — Công ty Bình Minh",
    mediaType: "audio",
    durationMs: 28 * 60 * 1000 + 40_000,
    createdAt: "2026-07-12T15:30:00",
    status: "done",
    docs: ["summary", "full_text"],
  },
  {
    id: "s-3",
    title: "Bài giảng: Kiến trúc hệ phân tán (buổi 4)",
    mediaType: "video",
    durationMs: 92 * 60 * 1000,
    createdAt: "2026-07-11T08:00:00",
    status: "done",
    docs: ["summary"],
  },
  {
    id: "s-4",
    title: "Ghi âm hiện trường — khảo sát nhà máy",
    mediaType: "audio",
    durationMs: 12 * 60 * 1000 + 5_000,
    createdAt: "2026-07-10T11:20:00",
    status: "failed",
    docs: [],
  },
  {
    id: "s-5",
    title: "Daily standup 12/07",
    mediaType: "audio",
    durationMs: 9 * 60 * 1000,
    createdAt: "2026-07-12T09:05:00",
    status: "done",
    docs: ["full_text"],
  },
];

export function getSource(id: string): Source | undefined {
  return MOCK_SOURCES.find((s) => s.id === id);
}

export const MOCK_TRANSCRIPT: TranscriptSegment[] = [
  {
    startMs: 0,
    endMs: 8_400,
    speaker: 0,
    text: "Chào cả nhà, mình bắt đầu buổi họp sprint nhé. Hôm nay có ba việc chính cần chốt.",
  },
  {
    startMs: 8_400,
    endMs: 19_200,
    speaker: 1,
    text: "Ừ, việc đầu tiên là phần realtime doc. Bên mình đã dựng xong khung WebSocket, giờ chờ pipeline cắt chunk.",
  },
  {
    startMs: 19_200,
    endMs: 31_000,
    speaker: 2,
    text: "Chunk theo ranh giới VAD thì mình xong phần proof of concept rồi, cắt 30 giây một, không đứt giữa câu.",
  },
  {
    startMs: 31_000,
    endMs: 44_500,
    speaker: 0,
    text: "Tốt. Vậy tuần này ưu tiên ghép hai phần đó lại. Việc thứ hai là ngân sách hạ tầng GPU cho quý ba.",
  },
  {
    startMs: 44_500,
    endMs: 58_800,
    speaker: 1,
    text: "Mình đề xuất giữ một GPU dùng chung, job xếp hàng tuần tự. Chưa cần chạy song song ở giai đoạn này.",
  },
  {
    startMs: 58_800,
    endMs: 69_000,
    speaker: 2,
    text: "Đồng ý. Vượt ngân sách nếu thuê thêm card, mà nhu cầu hiện tại chưa tới mức đó.",
  },
];

export const MOCK_SECTIONS: DocSection[] = [
  {
    id: "s1",
    heading: "Mục tiêu buổi họp",
    markdown:
      "Buổi họp Sprint 12 chốt **ba việc chính**: tiến độ tính năng tạo tài liệu thời gian thực, ngân sách hạ tầng GPU cho quý ba, và phân công tuần tới.",
  },
  {
    id: "s2",
    heading: "Tính năng tạo tài liệu thời gian thực",
    markdown:
      "Khung WebSocket đã hoàn thiện. Pipeline cắt audio **theo ranh giới VAD** (mỗi đoạn ~30 giây, không cắt giữa câu) đã có bản proof-of-concept. Ưu tiên tuần này là ghép hai phần lại để chạy end-to-end.",
  },
  {
    id: "s3",
    heading: "Ngân sách hạ tầng GPU — Quý 3",
    markdown:
      "Thống nhất giữ **một GPU dùng chung**, các job xử lý xếp hàng tuần tự. Chưa chạy song song và chưa thuê thêm card ở giai đoạn này để tránh vượt ngân sách.",
    writing: true,
  },
];
