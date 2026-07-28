"use client";

/**
 * GoogleLoginButton — nút đăng nhập bằng tài khoản Google.
 *
 * Sử dụng next-auth với Google Provider.
 * Trước khi dùng, cần cấu hình:
 *  1. Tạo Google OAuth credentials tại https://console.cloud.google.com
 *  2. Thêm vào .env.local:
 *       GOOGLE_CLIENT_ID=your_client_id
 *       GOOGLE_CLIENT_SECRET=your_client_secret
 *       NEXTAUTH_SECRET=any_random_string
 *       NEXTAUTH_URL=http://localhost:3000
 *  3. Tạo file app/api/auth/[...nextauth]/route.ts (xem hướng dẫn bên dưới)
 */

import { signIn, signOut, useSession } from "next-auth/react";
import { useState } from "react";

interface GoogleLoginButtonProps {
  /** Redirect về trang này sau khi đăng nhập thành công. Mặc định: "/" */
  callbackUrl?: string;
  /** Hiện ảnh đại diện và tên sau khi đăng nhập. Mặc định: true */
  showUserInfo?: boolean;
}

/** Icon Google SVG (không phụ thuộc thư viện icon ngoài). */
function GoogleIcon({ size = 20 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
        fill="#4285F4"
      />
      <path
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
        fill="#34A853"
      />
      <path
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"
        fill="#FBBC05"
      />
      <path
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
        fill="#EA4335"
      />
    </svg>
  );
}

export function GoogleLoginButton({
  callbackUrl = "/",
  showUserInfo = true,
}: GoogleLoginButtonProps) {
  const { data: session, status } = useSession();
  const [isLoading, setIsLoading] = useState(false);

  const isSessionLoading = status === "loading";
  const isLoggedIn = status === "authenticated";

  // --- Đã đăng nhập: hiển thị thông tin user + nút đăng xuất ---
  if (isLoggedIn && session?.user && showUserInfo) {
    return (
      <div className="flex items-center gap-3">
        {session.user.image && (
          <img
            src={session.user.image}
            alt={session.user.name ?? "Avatar"}
            width={36}
            height={36}
            className="rounded-full border border-line"
          />
        )}
        <div className="hidden sm:block">
          <p className="text-sm font-medium text-ink leading-tight">
            {session.user.name}
          </p>
          <p className="text-xs text-ink-faint truncate max-w-[160px]">
            {session.user.email}
          </p>
        </div>
        <button
          onClick={() => signOut({ callbackUrl: "/" })}
          className="ml-1 rounded-lg border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink-soft transition hover:bg-surface-2 hover:text-ink"
        >
          Đăng xuất
        </button>
      </div>
    );
  }

  // --- Chưa đăng nhập: nút đăng nhập Google ---
  async function handleSignIn() {
    setIsLoading(true);
    try {
      await signIn("google", { callbackUrl });
    } finally {
      // next-auth tự redirect nên finally chỉ là fallback
      setIsLoading(false);
    }
  }

  return (
    <button
      id="google-login-btn"
      type="button"
      onClick={handleSignIn}
      disabled={isLoading || isSessionLoading}
      className="
        inline-flex items-center gap-3
        rounded-xl border border-line bg-surface
        px-5 py-2.5
        text-sm font-medium text-ink
        shadow-sm
        transition-all duration-150
        hover:bg-surface-2 hover:shadow-md
        active:scale-[0.98]
        disabled:cursor-not-allowed disabled:opacity-60
      "
    >
      {isLoading || isSessionLoading ? (
        /* Spinner khi đang chuyển hướng */
        <svg
          className="h-5 w-5 animate-spin text-ink-faint"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
          />
        </svg>
      ) : (
        <GoogleIcon size={20} />
      )}
      <span>
        {isLoading ? "Đang chuyển hướng…" : "Đăng nhập bằng Google"}
      </span>
    </button>
  );
}
