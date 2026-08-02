/**
 * Middleware Next.js — bao ve cac route can dang nhap.
 *
 * Logic:
 *  1. /login va /api/auth/* luon cho phep (public).
 *  2. Neu co cookie guest_mode=true → cho qua (dung khong can dang nhap).
 *  3. Neu co session JWT → cho qua.
 *  4. Nguoc lai → redirect ve /login.
 */
import { getToken } from "next-auth/jwt";
import { NextRequest, NextResponse } from "next/server";

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Cac route luon public
  if (
    pathname.startsWith("/login") ||
    pathname.startsWith("/api/auth") ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon")
  ) {
    return NextResponse.next();
  }

  // Cho phep neu dang o che do khach (guest)
  const guestMode = req.cookies.get("guest_mode")?.value;
  if (guestMode === "true") {
    return NextResponse.next();
  }

  // Kiem tra session JWT cua next-auth
  const token = await getToken({
    req,
    secret: process.env.NEXTAUTH_SECRET ?? "change-me",
  });

  if (token) {
    return NextResponse.next();
  }

  // Chua dang nhap va khong phai khach → ve /login
  const loginUrl = new URL("/login", req.url);
  loginUrl.searchParams.set("callbackUrl", req.url);
  return NextResponse.redirect(loginUrl);
}

export const config = {
  // Ap dung cho tat ca route ngoai static files
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\..*).*)"],
};
