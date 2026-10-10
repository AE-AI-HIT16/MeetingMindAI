import type { NextConfig } from "next";
import path from "path";

// Proxy /v1/* sang FastAPI khi dev
const API = process.env.MEETASR_API ?? "http://127.0.0.1:8000";
const publicDevOrigin = (() => {
  try {
    return process.env.NEXTAUTH_URL ? new URL(process.env.NEXTAUTH_URL).hostname : null;
  } catch {
    return null;
  }
})();

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
};

export default nextConfig;
