"use client";

/**
 * AuthProvider — bọc toàn bộ app trong SessionProvider của next-auth.
 *
 * Phải là Client Component vì SessionProvider dùng React Context.
 * Được dùng trong layout.tsx để mọi component con đều gọi được
 * ``useSession()`` mà không cần truyền prop.
 */

import { SessionProvider } from "next-auth/react";
import type { ReactNode } from "react";

export function AuthProvider({ children }: { children: ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>;
}
