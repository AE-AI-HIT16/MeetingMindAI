import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    host: `${process.env.RUNPOD_ENDPOINT_ID}.api.runpod.ai`,
    token: process.env.RUNPOD_API_KEY,
  });
}
