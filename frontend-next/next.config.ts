import type { NextConfig } from "next";

// Proxy small API/media calls to FastAPI during dev. Large file uploads go
// directly to FastAPI because the Next proxy buffers and limits request bodies.
const API = process.env.MEETASR_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/v1/:path*", destination: `${API}/v1/:path*` }];
  },
};

export default nextConfig;
