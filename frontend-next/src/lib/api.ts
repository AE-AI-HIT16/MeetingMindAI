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

import type {
  CreateSourceResponse,
  DocMode,
  DocumentData,
  DocumentGeneration,
  Source,
} from "./types";

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
  input: string,
  init?: RequestInit,
  token?: string,
): Promise<T> {
  const isServer = typeof window === "undefined";
  let url: string;

  if (isServer) {
    const endpointId = process.env.RUNPOD_ENDPOINT_ID ?? "";
    const apiBase = `https://${endpointId}.api.runpod.ai`;
    url = new URL(input, apiBase).toString();
  } else {
    url = input;
  }
  
  const headers = new Headers(init?.headers);
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (isServer && process.env.RUNPOD_API_KEY) {
    headers.set("Authorization", `Bearer ${process.env.RUNPOD_API_KEY}`);
  }

  const res = await fetch(url, { ...init, headers });

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message =
        body?.detail ??
        body?.message ??
        body?.error?.message ??
        message;
    } catch {
      // bỏ qua nếu body không phải JSON
    }
    throw new APIError(res.status, message);
  }

  // 204 No Content — không có body
  if (res.status === 204) return undefined as T;

  return res.json() as Promise<T>;
}

/**
 * Browser operations that should not depend on the Next.js rewrite use the
 * public FastAPI origin. This covers large uploads and background finalize
 * submission/status calls.
 */
function directApiUrl(path: string): string {
  return `/api/runpod${path}`;
}

// ---------------------------------------------------------------------------
// Sources API
// ---------------------------------------------------------------------------

/**
 * Lấy danh sách tất cả Source (trang Library).
 * Tương đương: GET /v1/sources
 */
export async function listSources(token?: string): Promise<Source[]> {
  return apiFetch<Source[]>("/v1/sources", { cache: "no-store" }, token);
}

/**
 * Lấy chi tiết một Source theo ID.
 * Tương đương: GET /v1/sources/{id}
 */
export async function getSource(id: string, token?: string): Promise<Source> {
  return apiFetch<Source>(`/v1/sources/${id}`, { cache: "no-store" }, token);
}

/**
 * Upload file media — tạo Source mới và Job xử lý.
 * Tương đương: POST /v1/sources (multipart/form-data)
 *
 * @param file  File người dùng chọn từ <input> hoặc drag-and-drop.
 * @param onProgress  Callback nhận % upload (0–100); chỉ hoạt động trên trình duyệt.
 * @returns ID của Source và Job vừa tạo (status = "queued").
 */
import { getSession } from "next-auth/react";

export async function uploadSource(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<CreateSourceResponse> {
  const session = await getSession();
  const token = session?.accessToken;
  const formData = new FormData();
  formData.append("file", file);

  // Dùng XMLHttpRequest để có progress event (fetch không hỗ trợ upload progress)
  if (onProgress) {
    return new Promise<CreateSourceResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", directApiUrl("/v1/sources"));
      if (token) {
        xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      }

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as CreateSourceResponse);
          } catch {
            reject(new APIError(xhr.status, "Phản hồi không hợp lệ từ server."));
          }
        } else {
          let message = `HTTP ${xhr.status}`;
          try {
            const body = JSON.parse(xhr.responseText);
            message =
              body?.detail ??
              body?.message ??
              body?.error?.message ??
              message;
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
  return apiFetch<CreateSourceResponse>(directApiUrl("/v1/sources"), {
    method: "POST",
    body: formData,
  }, token);
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

// ---------------------------------------------------------------------------
// Documents API
// ---------------------------------------------------------------------------

export async function getDocument(id: string, token?: string): Promise<DocumentData> {
  return apiFetch<DocumentData>(`/v1/documents/${id}`, {
    cache: "no-store",
  }, token);
}

export async function finalizeDocument(
  liveDocumentId: string,
  mode: Exclude<DocMode, "live">,
): Promise<DocumentGeneration> {
  const generation = await apiFetch<DocumentGeneration>(
    directApiUrl(`/v1/documents/${liveDocumentId}/finalize`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    },
  );
  if (
    !generation?.generationJobId ||
    !generation.documentId ||
    !generation.status
  ) {
    throw new APIError(
      502,
      "Backend đang dùng contract finalize cũ. Hãy khởi động lại FastAPI rồi thử lại.",
    );
  }
  return generation;
}

export async function getDocumentGeneration(
  generationJobId: string,
): Promise<DocumentGeneration> {
  return apiFetch<DocumentGeneration>(
    directApiUrl(`/v1/document-jobs/${generationJobId}`),
    { cache: "no-store" },
  );
}

export function documentExportUrl(
  documentId: string,
  format: "md" | "docx" | "pdf",
): string {
  return `/v1/documents/${documentId}/export?format=${format}`;
}
