/**
 * next-auth route handler — xu ly OAuth flow cho Google va GitHub.
 *
 * Dat tai: app/api/auth/[...nextauth]/route.ts
 * URL tu dong: /api/auth/signin, /api/auth/callback/google,
 *              /api/auth/callback/github, ...
 *
 * Yeu cau bien moi truong trong .env.local:
 *   GOOGLE_CLIENT_ID      — tu Google Cloud Console
 *   GOOGLE_CLIENT_SECRET  — tu Google Cloud Console
 *   GITHUB_CLIENT_ID      — tu GitHub Developer Settings
 *   GITHUB_CLIENT_SECRET  — tu GitHub Developer Settings
 *   NEXTAUTH_SECRET       — chuoi ngau nhien bat ky
 *   NEXTAUTH_URL          — URL goc cua app (vd: http://localhost:3000)
 */

import { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import GitHubProvider from "next-auth/providers/github";

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID ?? "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET ?? "",
    }),
    GitHubProvider({
      clientId: process.env.GITHUB_CLIENT_ID ?? "",
      clientSecret: process.env.GITHUB_CLIENT_SECRET ?? "",
    }),
  ],

  pages: {
    signIn: "/login",
  },

  callbacks: {
    async jwt({ token, account, user }) {
      const apiBase = process.env.MEETASR_API || "http://127.0.0.1:8000";

      // --- Lần đăng nhập đầu tiên: Sync với Backend ---
      if (account && user) {
        try {
          const res = await fetch(`${apiBase}/v1/auth/sync`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              provider: account.provider,
              provider_id: account.providerAccountId,
              email: user.email || "",
              name: user.name || "",
              avatar_url: user.image || null,
              sync_secret: process.env.NEXTAUTH_SECRET || "",
            }),
          });

          if (res.ok) {
            const data = await res.json();
            token.accessToken = data.access_token;
            token.sub = data.user.id;
            // Lưu thời điểm hết hạn (expires_in trả về = giây)
            token.tokenExpiry = Date.now() + data.expires_in * 1000;
          } else {
            console.error("Backend sync failed", await res.text());
          }
        } catch (error) {
          console.error("Lỗi khi gọi /v1/auth/sync:", error);
        }
        return token;
      }

      // --- Các lần tiếp theo: Auto-refresh nếu còn < 7 ngày ---
      const sevenDaysMs = 7 * 24 * 60 * 60 * 1000;
      const expiry = (token.tokenExpiry as number | undefined) ?? 0;
      const shouldRefresh = expiry - Date.now() < sevenDaysMs;

      if (token.accessToken && shouldRefresh) {
        try {
          const res = await fetch(`${apiBase}/v1/auth/refresh`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token.accessToken}`,
            },
          });

          if (res.ok) {
            const data = await res.json();
            token.accessToken = data.access_token;
            token.tokenExpiry = Date.now() + data.expires_in * 1000;
            console.log("✅ Token đã được gia hạn tự động.");
          } else {
            // Token đã hết hạn hoàn toàn, xóa để buộc đăng nhập lại
            console.warn("⚠️ Refresh thất bại, yêu cầu đăng nhập lại.");
            token.accessToken = undefined;
            token.tokenExpiry = undefined;
          }
        } catch (error) {
          console.error("Lỗi khi gọi /v1/auth/refresh:", error);
        }
      }

      return token;
    },

    /** Truyền thêm thông tin user và JWT vào session. */
    async session({ session, token }) {
      if (session.user && token.sub) {
        (session.user as { id?: string }).id = token.sub;
      }
      // Gắn JWT nội bộ vào session để client gọi API
      (session as any).accessToken = token.accessToken;
      return session;
    },
  },
};

import NextAuth from "next-auth";
const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
