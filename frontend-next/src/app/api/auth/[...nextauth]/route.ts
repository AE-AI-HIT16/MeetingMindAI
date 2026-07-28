/**
 * next-auth route handler — xử lý toàn bộ OAuth flow của Google.
 *
 * Đặt tại: app/api/auth/[...nextauth]/route.ts
 * URL tự động: /api/auth/signin, /api/auth/callback/google, ...
 *
 * Yêu cầu các biến môi trường trong .env.local:
 *   GOOGLE_CLIENT_ID      — từ Google Cloud Console
 *   GOOGLE_CLIENT_SECRET  — từ Google Cloud Console
 *   NEXTAUTH_SECRET       — chuỗi ngẫu nhiên bất kỳ (dùng: openssl rand -base64 32)
 *   NEXTAUTH_URL          — URL gốc của app (vd: http://localhost:3000)
 */

import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";

const handler = NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID ?? "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET ?? "",
    }),
  ],

  // Tuỳ chọn: trang đăng nhập mặc định của next-auth
  pages: {
    signIn: "/login", // redirect về /login thay vì trang mặc định của next-auth
  },

  callbacks: {
    /** Truyền thêm thông tin user vào session nếu cần. */
    async session({ session, token }) {
      if (session.user && token.sub) {
        (session.user as { id?: string }).id = token.sub;
      }
      return session;
    },
  },
});

export { handler as GET, handler as POST };
