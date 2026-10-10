import type { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    accessToken?: string;
    /** Set when the session has no backend JWT (see the session callback). */
    authError?: "syncing" | "relogin";
    user?: DefaultSession["user"] & {
      id?: string;
    };
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
    tokenExpiry?: number;
  }
}
