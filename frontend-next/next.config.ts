import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  // Bao Turbopack biet dung root la thu muc frontend-next, tranh nham voi lockfile o thu muc cha
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
