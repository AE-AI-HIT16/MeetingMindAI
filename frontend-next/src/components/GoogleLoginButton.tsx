"use client";

/**
 * GoogleLoginButton — nut dang nhap bang Google va GitHub.
 *
 * Su dung next-auth voi Google Provider va GitHub Provider.
 * Can cau hinh .env.local:
 *   GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
 *   GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET
 *   NEXTAUTH_SECRET / NEXTAUTH_URL
 */

import { signIn, signOut, useSession } from "next-auth/react";
import Image from "next/image";
import { useState } from "react";

interface GoogleLoginButtonProps {
  /** Redirect ve trang nay sau khi dang nhap thanh cong. Mac dinh: "/" */
  callbackUrl?: string;
  /** Hien anh dai dien va ten sau khi dang nhap. Mac dinh: true */
  showUserInfo?: boolean;
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

function GoogleIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05" />
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
    </svg>
  );
}

function GitHubIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.39.6.11.82-.26.82-.58 0-.28-.01-1.02-.02-2C5.67 21.47 4.97 19.23 4.97 19.23c-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.08 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.5.99.11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.005 2.04.14 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.24 2.87.12 3.17.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.62-5.48 5.92.43.37.81 1.1.81 2.22 0 1.6-.01 2.89-.01 3.28 0 .32.21.7.82.58C20.56 21.79 24 17.3 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Component chinh
// ---------------------------------------------------------------------------

export function GoogleLoginButton({
  callbackUrl = "/",
  showUserInfo = true,
}: GoogleLoginButtonProps) {
  const { data: session, status } = useSession();
  const [loadingProvider, setLoadingProvider] = useState<"google" | "github" | null>(null);

  const isSessionLoading = status === "loading";
  const isLoggedIn = status === "authenticated";

  // --- Da dang nhap: hien anh dai dien + ten + nut dang xuat ---
  if (isLoggedIn && session?.user && showUserInfo) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2.5">
          {session.user.image && (
            <Image
              src={session.user.image}
              alt={session.user.name ?? "Avatar"}
              width={32}
              height={32}
              unoptimized
              className="rounded-full border border-line shrink-0"
            />
          )}
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-ink leading-tight truncate">
              {session.user.name}
            </p>
            <p className="text-xs text-ink-faint truncate">
              {session.user.email}
            </p>
          </div>
        </div>
        <button
          id="signout-btn"
          onClick={() => signOut({ callbackUrl: "/" })}
          className="w-full rounded-lg border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink-soft transition hover:bg-surface-2 hover:text-ink"
        >
          Dang xuat
        </button>
      </div>
    );
  }

  // --- Chua dang nhap: 2 nut Google + GitHub ---
  async function handleSignIn(provider: "google" | "github") {
    setLoadingProvider(provider);
    try {
      await signIn(provider, { callbackUrl });
    } finally {
      setLoadingProvider(null);
    }
  }

  const isDisabled = !!loadingProvider || isSessionLoading;

  return (
    <div className="flex flex-col gap-2">
      {/* Nut Google */}
      <button
        id="google-login-btn"
        type="button"
        onClick={() => handleSignIn("google")}
        disabled={isDisabled}
        className="inline-flex w-full items-center justify-center gap-2.5 rounded-xl border border-line bg-surface px-4 py-2.5 text-sm font-medium text-ink shadow-sm transition-all hover:bg-surface-2 hover:shadow-md active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loadingProvider === "google" ? <Spinner /> : <GoogleIcon />}
        <span>Dang nhap bang Google</span>
      </button>

      {/* Nut GitHub */}
      <button
        id="github-login-btn"
        type="button"
        onClick={() => handleSignIn("github")}
        disabled={isDisabled}
        className="inline-flex w-full items-center justify-center gap-2.5 rounded-xl border border-line bg-[#24292e] px-4 py-2.5 text-sm font-medium text-white shadow-sm transition-all hover:bg-[#1a1f24] hover:shadow-md active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loadingProvider === "github" ? <Spinner /> : <GitHubIcon />}
        <span>Dang nhap bang GitHub</span>
      </button>
    </div>
  );
}
