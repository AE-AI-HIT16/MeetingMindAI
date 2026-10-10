"use client";

/**
 * AuthProvider — bọc toàn bộ app trong SessionProvider của next-auth.
 *
 * Phải là Client Component vì SessionProvider dùng React Context.
 * Được dùng trong layout.tsx để mọi component con đều gọi được
 * ``useSession()`` mà không cần truyền prop.
 */

import { SessionProvider, signOut, useSession } from "next-auth/react";
import { useEffect, type ReactNode } from "react";

export function AuthProvider({ children }: { children: ReactNode }) {
  return (
    <SessionProvider>
      <StaleSessionGuard />
      {children}
    </SessionProvider>
  );
}

/** Sessions that can never get a backend token are signed out, so the user
 *  logs in again instead of silently creating ownerless uploads. */
function StaleSessionGuard() {
  const { data: session } = useSession();
  const relogin = session?.authError === "relogin";
  useEffect(() => {
    if (relogin) void signOut({ callbackUrl: "/login" });
  }, [relogin]);
  return null;
}
