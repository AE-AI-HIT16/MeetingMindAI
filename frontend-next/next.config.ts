import type { NextConfig } from "next";
import path from "path";

// Proxy /v1/* sang FastAPI khi dev
const API = process.env.MEETASR_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [{ source: "/v1/:path*", destination: `${API}/v1/:path*` }];
  },
  // Bao Turbopack biet dung root la thu muc frontend-next, tranh nham voi lockfile o thu muc cha
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
