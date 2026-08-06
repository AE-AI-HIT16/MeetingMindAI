import type { NextConfig } from "next";
import path from "path";

// Proxy /v1/* sang local API proxy
const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [{ source: "/v1/:path*", destination: "/api/runpod/v1/:path*" }];
  },
  // Bao Turbopack biet dung root la thu muc frontend-next, tranh nham voi lockfile o thu muc cha
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
