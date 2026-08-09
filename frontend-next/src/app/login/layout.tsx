/**
 * Layout rieng cho trang /login — khong hien Sidebar.
 * Next.js tu dong ap dung layout nay cho tat ca route trong /login/.
 */
import type { ReactNode } from "react";

export default function LoginLayout({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
