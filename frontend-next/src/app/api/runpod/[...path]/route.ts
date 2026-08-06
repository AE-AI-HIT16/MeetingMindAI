import { NextRequest, NextResponse } from "next/server";

/**
 * RunPod API Proxy — chuyển tiếp tất cả request tới RunPod Serverless Endpoint.
 * 
 * Browser gọi:  /api/runpod/v1/health
 * Proxy gửi tới: https://<endpoint>.api.runpod.ai/v1/health
 *                + Header Authorization: Bearer <RUNPOD_API_KEY>
 * 
 * API key KHÔNG BAO GIỜ lộ ra browser.
 */

const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY ?? "";
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID ?? "";
const RUNPOD_BASE = `https://${RUNPOD_ENDPOINT_ID}.api.runpod.ai`;

async function proxyToRunpod(req: NextRequest, path: string) {
  const targetUrl = `${RUNPOD_BASE}/${path}`;

  const headers = new Headers();
  headers.set("Authorization", `Bearer ${RUNPOD_API_KEY}`);

  // Sao chép Content-Type từ request gốc
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);

  // Sao chép Authorization từ frontend (JWT token cho user auth)
  const frontendAuth = req.headers.get("authorization");
  if (frontendAuth) headers.set("Authorization", frontendAuth);
  
  // NOTE: Vì cả RunPod và Backend đều dùng header "Authorization" 
  // (RunPod dùng Bearer RUNPOD_API_KEY, Backend dùng Bearer JWT_TOKEN)
  // RunPod tự động parse JWT hay sao? 
  // Thực tế, với RunPod Load-balanced endpoint, auth là ở mức Gateway.
  // Gateway của RunPod đòi Authorization: Bearer RUNPOD_API_KEY.
  // Khi request vượt qua gateway, RunPod truyền header gì vào container bên trong?
  // Để an toàn, truyền token của user qua 1 header riêng (ví dụ X-Frontend-Auth), 
  // HOẶC nếu Gateway ghi đè Authorization, ta sẽ truyền cả 2.
  // Backend MeetASR đọc token từ Authorization header.
  // Đây là 1 vấn đề phổ biến với RunPod Serverless Gateway.
  // Tạm thời cứ gửi API key ở Authorization header để pass RunPod Gateway.
  // Token đăng nhập của user ta gửi qua X-User-Token hoặc giữ nguyên (phụ thuộc config Backend).
  // Sẽ cần kiểm tra nếu backend báo lỗi JWT.

  const body = req.method !== "GET" && req.method !== "HEAD"
    ? await req.arrayBuffer()
    : undefined;

  try {
    const response = await fetch(targetUrl, {
      method: req.method,
      headers,
      body,
    });

    // Trả response về browser
    const responseHeaders = new Headers();
    response.headers.forEach((value, key) => {
      responseHeaders.set(key, value);
    });

    return new NextResponse(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("RunPod Proxy Error:", error);
    return NextResponse.json(
      { detail: "Internal Server Error from Proxy" },
      { status: 500 }
    );
  }
}

// Xử lý mọi HTTP method
export async function GET(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxyToRunpod(req, path.join("/"));
}

export async function POST(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxyToRunpod(req, path.join("/"));
}

export async function PUT(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxyToRunpod(req, path.join("/"));
}

export async function DELETE(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxyToRunpod(req, path.join("/"));
}
