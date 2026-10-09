import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    wsBase: process.env.MEETASR_API?.replace(/^http:/, "ws:").replace(/^https:/, "wss:") ?? "ws://127.0.0.1:8000",
  });
}
