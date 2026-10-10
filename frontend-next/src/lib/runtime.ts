export function getApiBase(): string {
  let base =
    process.env.NEXT_PUBLIC_MEETASR_API ||
    process.env.MEETASR_API ||
    "http://127.0.0.1:8000";
  base = base.trim().replace(/^['"]|['"]$/g, "").replace(/\/+$/, "");
  if (!base.startsWith("http://") && !base.startsWith("https://")) {
    base = `https://${base}`;
  }
  return base;
}

export function getWebSocketBase(): string {
  return getApiBase().replace(/^http:/, "ws:").replace(/^https:/, "wss:");
}
