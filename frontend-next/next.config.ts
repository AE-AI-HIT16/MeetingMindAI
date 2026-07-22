import type { NextConfig } from "next";

// Proxy API + WebSocket calls to the FastAPI backend during dev,
// so the frontend can call `/v1/...` with no CORS setup.
const API = process.env.MEETASR_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/v1/:path*", destination: `${API}/v1/:path*` }];
  },
};

export default nextConfig;
