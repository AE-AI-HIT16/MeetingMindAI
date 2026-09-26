export interface DesktopRuntime {
  isDesktop?: boolean;
  apiBase?: string;
}

declare global {
  interface Window {
    __MEETASR_API_BASE__?: string;
    meetasrDesktop?: DesktopRuntime;
  }
}

export function getApiBase(): string {
  const runtimeBase =
    typeof window !== "undefined"
      ? window.__MEETASR_API_BASE__ || window.meetasrDesktop?.apiBase
      : undefined;
  let base =
    runtimeBase ||
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
