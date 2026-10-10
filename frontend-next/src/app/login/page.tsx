"use client";

import { signIn, useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  CircleNotch,
  FileText,
  GithubLogo,
  UserCircleDashed,
  UsersThree,
  Waveform as WaveformIcon,
  type Icon,
} from "@phosphor-icons/react";
import { Logo } from "@/components/Sidebar";
import { Waveform } from "@/components/ui";

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

function Spinner() {
  return <CircleNotch size={20} className="animate-spin" />;
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

  // Soft spotlight that trails the pointer. It is a fixed-size layer moved
  // with transform only (compositor, no repaint, no React re-render).
  const spotRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const spot = spotRef.current;
    const finePointer = window.matchMedia("(pointer: fine)").matches;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!spot || !finePointer || reduce) return;
    let frame = 0;
    const onMove = (e: PointerEvent) => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        spot.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
        spot.style.opacity = "1";
      });
    };
    window.addEventListener("pointermove", onMove);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", onMove);
    };
  }, []);

  return (
    <div className="relative isolate min-h-dvh overflow-hidden">
      {/* Background, all decorative: drifting light pools, a dot grid that
          fades out from the center, a live waveform horizon, a pointer
          spotlight. Grain comes from body::after. */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
        <div className="drift absolute -left-40 -top-48 h-[38rem] w-[38rem] rounded-full bg-[radial-gradient(circle,rgb(224_85_47/0.24),transparent_65%)]" />
        <div className="drift-slow absolute -bottom-56 -right-40 h-[44rem] w-[44rem] rounded-full bg-[radial-gradient(circle,rgb(59_79_216/0.18),transparent_65%)]" />
        <div className="drift absolute left-[38%] top-[30%] h-[26rem] w-[26rem] rounded-full bg-[radial-gradient(circle,rgb(242_176_102/0.18),transparent_65%)] [animation-delay:-9s]" />
        <div className="dot-grid absolute inset-0 [mask-image:radial-gradient(ellipse_70%_60%_at_50%_45%,black,transparent)]" />
        <div ref={spotRef} className="spotlight fixed left-0 top-0 opacity-0 transition-opacity duration-700" />
        <div className="absolute inset-x-0 bottom-0 h-48 opacity-25 [mask-image:linear-gradient(to_top,black,transparent)]">
          <Waveform live bars={64} className="h-full w-full" />
        </div>
      </div>

      <div className="mx-auto grid min-h-dvh w-full max-w-6xl items-center gap-12 px-4 py-12 sm:px-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,0.85fr)] lg:gap-16 lg:px-10">
        {/* Story */}
        <section className="animate-enter">
          <div className="flex items-center gap-2.5">
            <Logo size={36} />
            <span className="font-display text-xl font-semibold tracking-tight text-ink">
              MeetingMind
            </span>
          </div>

          {/* Lục bát couplet: "ga" rhymes with "ra". */}
          <h1 className="mt-14 font-display text-[34px] font-semibold leading-[1.12] tracking-[-0.035em] text-ink sm:text-[42px] lg:text-[34px] xl:text-[40px]">
            <span className="block">Họp hành cứ nói thả ga.</span>
            <span className="block text-ink-faint sm:whitespace-nowrap">
              Biên bản <span className="text-brand">để máy</span>, sếp ra là xong.
            </span>
          </h1>
          <p className="mt-6 max-w-md text-lg leading-relaxed text-ink-soft">
            Ai nói gì, chốt việc gì, ai làm tiếp. MeetingMind ghi đủ cả.
          </p>

          <ul className="mt-12 hidden gap-x-8 gap-y-5 sm:grid sm:grid-cols-3 lg:max-w-xl">
            {FEATURES.map(({ icon: FeatureIcon, title, desc }) => (
              <li key={title}>
                <FeatureIcon size={22} className="text-ink" />
                <p className="mt-3 text-sm font-semibold text-ink">{title}</p>
                <p className="mt-1 text-[13px] leading-relaxed text-ink-faint">{desc}</p>
              </li>
            ))}
          </ul>
        </section>

        {/* Sign-in */}
        <section className="animate-enter w-full max-w-md justify-self-center [animation-delay:120ms] lg:justify-self-end">
          <div className="relative">
          {/* Slow aurora glowing around the card edges */}
          <div aria-hidden="true" className="aurora pointer-events-none absolute -inset-6 -z-10 rounded-[3rem] opacity-50" />
          <div className="bezel bg-white/40 backdrop-blur-xl">
            <div className="bezel-core px-7 py-9 sm:px-9">
              <h2 className="font-display text-2xl font-semibold tracking-tight text-ink">
                Chào mừng trở lại
              </h2>
              <p className="mt-1.5 text-sm text-ink-soft">
                Đăng nhập để lưu lịch sử và tài liệu của bạn.
              </p>

              <div className="mt-8 space-y-3">
                {/* Google */}
                <button
                  id="google-login-btn"
                  type="button"
                  onClick={() => handleSignIn("google")}
                  disabled={isDisabled}
                  className="btn btn-primary w-full py-3"
                >
                  {loading === "google" ? <Spinner /> : <GoogleIcon />}
                  <span>Tiếp tục bằng Google</span>
                </button>

                {/* GitHub */}
                <button
                  id="github-login-btn"
                  type="button"
                  onClick={() => handleSignIn("github")}
                  disabled={isDisabled}
                  className="btn btn-secondary w-full py-3"
                >
                  {loading === "github" ? <Spinner /> : <GithubLogo size={20} weight="fill" />}
                  <span>Tiếp tục bằng GitHub</span>
                </button>
              </div>

              {/* Divider */}
              <div className="my-7 flex items-center gap-3">
                <div className="h-px flex-1 bg-line" />
                <span className="text-xs text-ink-faint">hoặc</span>
                <div className="h-px flex-1 bg-line" />
              </div>

              {/* Guest */}
              <button
                id="guest-btn"
                type="button"
                onClick={handleGuest}
                disabled={isDisabled}
                className="group flex w-full items-center justify-between gap-3 rounded-2xl bg-surface-2 px-4 py-3.5 text-left ring-1 ring-line-soft transition hover:bg-surface-3 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <span className="flex items-center gap-3">
                  {loading === "guest" ? <Spinner /> : <UserCircleDashed size={22} className="text-ink-soft" />}
                  <span>
                    <span className="block text-sm font-semibold text-ink">Dùng thử không cần tài khoản</span>
                    <span className="block text-xs text-ink-faint">Lịch sử không được lưu khi đóng trình duyệt.</span>
                  </span>
                </span>
                <ArrowRight size={16} className="shrink-0 text-ink-faint transition group-hover:translate-x-0.5 group-hover:text-ink" />
              </button>
            </div>
          </div>

          </div>
          <p className="mt-6 text-center text-xs text-ink-faint">
            © {new Date().getFullYear()} MeetingMind
          </p>
        </section>
      </div>
    </div>
  );
}

const FEATURES: { icon: Icon; title: string; desc: string }[] = [
  {
    icon: WaveformIcon,
    title: "Ghi âm hoặc tải lên",
    desc: "Micro, âm thanh từ tab, hoặc video/audio đến 2 GB.",
  },
  {
    icon: UsersThree,
    title: "Tự tách người nói",
    desc: "Mỗi người một màu, đánh dấu chỗ nói chồng.",
  },
  {
    icon: FileText,
    title: "Tóm tắt hoặc toàn văn",
    desc: "Xuất PDF, DOCX hoặc Markdown chỉ với một lần bấm.",
  },
];
