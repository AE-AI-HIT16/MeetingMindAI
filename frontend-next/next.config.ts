import type { NextConfig } from "next";
import path from "path";

// Proxy /v1/* sang FastAPI khi dev
const API = process.env.MEETASR_API ?? "http://127.0.0.1:8000";
const publicDevOrigin = process.env.NEXTAUTH_URL
  ? new URL(process.env.NEXTAUTH_URL).hostname
  : null;

const nextConfig: NextConfig = {
  allowedDevOrigins: [
    "127.0.0.1",
    "localhost",
    ...(publicDevOrigin ? [publicDevOrigin] : []),
  ],
  async rewrites() {
    return [{ source: "/v1/:path*", destination: `${API}/v1/:path*` }];
  },
  turbopack: {
    root: path.resolve(__dirname),
  },
  // Standalone output cho Electron production: tạo .next/standalone/server.js
  // tự chứa Node.js server không cần node_modules ngoài.
  output: process.env.ELECTRON_BUILD === "1" ? "standalone" : undefined,
};

export default nextConfig;
