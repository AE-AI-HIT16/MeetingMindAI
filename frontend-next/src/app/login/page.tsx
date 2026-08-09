"use client";

import { signIn, useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

function GoogleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05" />
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
    </svg>
  );
}

function GitHubIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.39.6.11.82-.26.82-.58 0-.28-.01-1.02-.02-2C5.67 21.47 4.97 19.23 4.97 19.23c-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.08 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.5.99.11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.005 2.04.14 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.24 2.87.12 3.17.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.62-5.48 5.92.43.37.81 1.1.81 2.22 0 1.6-.01 2.89-.01 3.28 0 .32.21.7.82.58C20.56 21.79 24 17.3 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}

export default function LoginPage() {
  const { status } = useSession();
  const router = useRouter();
  const [loading, setLoading] = useState<"google" | "github" | "guest" | null>(null);

  // Da dang nhap → ve trang chu
  useEffect(() => {
    if (status === "authenticated") router.replace("/");
  }, [status, router]);

  async function handleSignIn(provider: "google" | "github") {
    setLoading(provider);
    await signIn(provider, { callbackUrl: "/" });
    setLoading(null);
  }

  function handleGuest() {
    setLoading("guest");
    // Dat cookie guest_mode — middleware cho qua
    document.cookie = "guest_mode=true; path=/; max-age=86400"; // het han sau 24h
    window.location.href = "/";
  }

  const isDisabled = !!loading || status === "loading";

  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-[#f8f7f4]">
      {/* Background gradient */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 60% at 50% -10%, rgba(99,102,241,0.12) 0%, transparent 70%)",
        }}
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute bottom-0 left-0 right-0 h-64"
        style={{
          background:
            "radial-gradient(ellipse 60% 50% at 50% 110%, rgba(99,102,241,0.08) 0%, transparent 70%)",
        }}
      />

      <div className="relative z-10 w-full max-w-sm px-6">
        {/* Logo + Brand */}
        <div className="mb-10 flex flex-col items-center gap-4 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#3d3ef7] shadow-lg shadow-indigo-200">
            <svg width="28" height="28" viewBox="0 0 18 18" fill="none">
              <rect x="1" y="7" width="2" height="4" rx="1" fill="white" />
              <rect x="5" y="3" width="2" height="12" rx="1" fill="white" />
              <rect x="9" y="5" width="2" height="8" rx="1" fill="white" />
              <rect x="13" y="8" width="2" height="2" rx="1" fill="#FF6B4A" />
            </svg>
          </div>
          <div>
            <h1 className="font-display text-2xl font-semibold text-[#111]">
              MeetingMind
            </h1>
            <p className="mt-1 text-sm text-[#888]">
              Tu giong noi thanh tai lieu
            </p>
          </div>
        </div>

        {/* Card dang nhap */}
        <div className="rounded-2xl border border-[#e8e6e0] bg-white px-8 py-8 shadow-sm shadow-black/[0.04]">
          <h2 className="mb-1 text-center text-[15px] font-semibold text-[#111]">
            Chao mung ban tro lai
          </h2>
          <p className="mb-7 text-center text-xs text-[#999]">
            Dang nhap de luu lich su va tai lieu cua ban
          </p>

          {/* Google */}
          <button
            id="google-login-btn"
            type="button"
            onClick={() => handleSignIn("google")}
            disabled={isDisabled}
            className="mb-3 flex w-full items-center justify-center gap-3 rounded-xl border border-[#e2e0da] bg-white px-5 py-3 text-sm font-medium text-[#222] shadow-[0_1px_2px_rgba(0,0,0,0.06)] transition-all hover:bg-[#fafaf8] hover:shadow-md active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading === "google" ? <Spinner /> : <GoogleIcon />}
            <span>Tiep tuc bang Google</span>
          </button>

          {/* GitHub */}
          <button
            id="github-login-btn"
            type="button"
            onClick={() => handleSignIn("github")}
            disabled={isDisabled}
            className="flex w-full items-center justify-center gap-3 rounded-xl bg-[#24292e] px-5 py-3 text-sm font-medium text-white shadow-[0_1px_2px_rgba(0,0,0,0.2)] transition-all hover:bg-[#1a1f24] hover:shadow-md active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading === "github" ? <Spinner /> : <GitHubIcon />}
            <span>Tiep tuc bang GitHub</span>
          </button>

          {/* Divider */}
          <div className="my-5 flex items-center gap-3">
            <div className="h-px flex-1 bg-[#eee]" />
            <span className="text-xs text-[#bbb]">hoac</span>
            <div className="h-px flex-1 bg-[#eee]" />
          </div>

          {/* Guest */}
          <button
            id="guest-btn"
            type="button"
            onClick={handleGuest}
            disabled={isDisabled}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-[#ddd] px-5 py-2.5 text-sm text-[#888] transition-all hover:border-[#bbb] hover:text-[#555] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading === "guest" ? (
              <Spinner />
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            )}
            <span>Su dung khong can dang nhap</span>
          </button>

          {/* Note */}
          <p className="mt-4 text-center text-[11px] leading-relaxed text-[#bbb]">
            Su dung khach se khong luu lich su sau khi dong trinh duyet.
          </p>
        </div>

        <p className="mt-6 text-center text-xs text-[#bbb]">
          © {new Date().getFullYear()} MeetingMind
        </p>
      </div>
    </div>
  );
}
