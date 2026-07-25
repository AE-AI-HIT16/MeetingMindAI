/**
 * API client — tầng trung gian giữa frontend và FastAPI backend.
 *
 * Mọi lời gọi HTTP đều đi qua đây để:
 *  - Không hardcode URL trong từng component.
 *  - Dễ mock khi test.
 *  - Xử lý lỗi thống nhất (throw APIError với code + message).
 *
 * Next.js rewrite trong next.config.ts tự proxy /v1/* → http://127.0.0.1:8000/v1/*
 * nên không cần CORS setup và không cần biết host của backend.
 */

import type { Source } from "./types";

// ---------------------------------------------------------------------------
// Lỗi API có cấu trúc
// ---------------------------------------------------------------------------

export class APIError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "APIError";
  }
}

/** Gửi request và throw APIError nếu response không OK. */
async function apiFetch<T>(
  input: RequestInfo,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(input, init);

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = body?.detail ?? body?.message ?? message;
    } catch {
      // bỏ qua nếu body không phải JSON
    }
    throw new APIError(res.status, message);
  }

  // 204 No Content — không có body
  if (res.status === 204) return undefined as T;

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Sources API
// ---------------------------------------------------------------------------

/**
 * Lấy danh sách tất cả Source (trang Library).
 * Tương đương: GET /v1/sources
 */
export async function listSources(): Promise<Source[]> {
  return apiFetch<Source[]>("/v1/sources");
}

/**
 * Lấy chi tiết một Source theo ID.
 * Tương đương: GET /v1/sources/{id}
 */
export async function getSource(id: string): Promise<Source> {
  return apiFetch<Source>(`/v1/sources/${id}`);
}

/**
 * Upload file media — tạo Source mới và Job xử lý.
 * Tương đương: POST /v1/sources (multipart/form-data)
 *
 * @param file  File người dùng chọn từ <input> hoặc drag-and-drop.
 * @param onProgress  Callback nhận % upload (0–100); chỉ hoạt động trên trình duyệt.
 * @returns Source vừa tạo (status = "processing").
 */
export async function uploadSource(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<Source> {
  const formData = new FormData();
  formData.append("file", file);

  // Dùng XMLHttpRequest để có progress event (fetch không hỗ trợ upload progress)
  if (onProgress) {
    return new Promise<Source>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/v1/sources");

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as Source);
          } catch {
            reject(new APIError(xhr.status, "Phản hồi không hợp lệ từ server."));
          }
        } else {
          let message = `HTTP ${xhr.status}`;
          try {
            const body = JSON.parse(xhr.responseText);
            message = body?.detail ?? body?.message ?? message;
          } catch { /* bỏ qua */ }
          reject(new APIError(xhr.status, message));
        }
      });

      xhr.addEventListener("error", () =>
        reject(new APIError(0, "Lỗi kết nối mạng.")),
      );

      xhr.send(formData);
    });
  }

  // Không cần progress → dùng fetch đơn giản hơn
  return apiFetch<Source>("/v1/sources", { method: "POST", body: formData });
}

/**
 * Xóa một Source cùng toàn bộ dữ liệu liên quan.
 * Tương đương: DELETE /v1/sources/{id}
 */
export async function deleteSource(id: string): Promise<void> {
  await apiFetch<void>(`/v1/sources/${id}`, { method: "DELETE" });
}

/**
 * Trả về URL để phát file media trong thẻ <video> / <audio>.
 * Backend hỗ trợ HTTP Range nên trình duyệt có thể tua tự nhiên.
 */
export function mediaUrl(sourceId: string): string {
  return `/v1/sources/${sourceId}/media`;
}
