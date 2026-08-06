"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";

const NAV = [
  { href: "/", label: "Thư viện", icon: LibraryIcon },
  { href: "/upload", label: "Tải lên", icon: UploadIcon },
  { href: "/realtime", label: "Ghi âm trực tiếp", icon: MicIcon },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-line bg-surface px-5 py-6 md:flex">
      {/* Wordmark — the transformation, spelled out */}
      <Link href="/" className="mb-9 flex items-center gap-2.5">
        <Logo />
        <span className="font-display text-[19px] font-semibold text-ink">
          MeetingMind
        </span>
      </Link>

      <Link
        href="/upload"
        className="mb-8 flex items-center justify-center gap-2 rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white shadow-[0_1px_2px_rgb(30_43_166/0.4)] transition hover:bg-brand-ink"
      >
        <PlusIcon />
        Tài liệu mới
      </Link>

      <nav className="flex flex-col gap-1">
        <p className="eyebrow mb-2 px-3">Điều hướng</p>
        {NAV.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
                active
                  ? "bg-brand-wash text-brand-ink"
                  : "text-ink-soft hover:bg-surface-2 hover:text-ink"
              }`}
            >
              <Icon />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* User info hoac nut dang nhap */}
      <UserSection />

      {/* Storage */}
      <div className="mt-4 rounded-xl border border-line bg-surface-2 p-4">
        <div className="flex items-center justify-between">
          <p className="eyebrow">Dung luong</p>
          <span className="font-mono text-xs text-ink-soft">6.2 / 20 GB</span>
        </div>
        <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-line">
          <div className="h-full w-[31%] rounded-full bg-brand" />
        </div>
        <p className="mt-2 text-xs text-ink-faint">
          Audio, video va tai lieu duoc luu tren may chu.
        </p>
      </div>
    </aside>
  );
}

function UserSection() {
  const { data: session, status } = useSession();

  if (status === "loading") return null;

  if (status === "authenticated" && session?.user) {
    return (
      <div className="mt-4 flex items-center gap-2.5 rounded-xl border border-line bg-surface-2 p-3">
        {session.user.image ? (
          <Image
            src={session.user.image}
            alt={session.user.name ?? "Avatar"}
            width={32}
            height={32}
            unoptimized
            className="shrink-0 rounded-full border border-line"
          />
        ) : (
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand text-xs font-semibold text-white">
            {session.user.name?.[0]?.toUpperCase() ?? "U"}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-ink leading-tight">
            {session.user.name}
          </p>
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className="text-xs text-ink-faint transition hover:text-red-500"
          >
            Dang xuat
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-4">
      <Link
        href="/login"
        className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-dashed border-line px-4 py-2 text-xs text-ink-faint transition hover:border-brand hover:text-brand"
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
          <polyline points="10 17 15 12 10 7" />
          <line x1="15" y1="12" x2="3" y2="12" />
        </svg>
        Dang nhap de luu du lieu
      </Link>
    </div>
  );
}

function Logo() {
  // A sound mark resolving into a baseline — voice → text.
  return (
    <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand">
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <rect x="1" y="7" width="2" height="4" rx="1" fill="white" />
        <rect x="5" y="3" width="2" height="12" rx="1" fill="white" />
        <rect x="9" y="5" width="2" height="8" rx="1" fill="white" />
        <rect x="13" y="8" width="2" height="2" rx="1" fill="#FF6B4A" />
      </svg>
    </span>
  );
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}
function LibraryIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="7" height="16" rx="1.5" />
      <rect x="14" y="4" width="7" height="16" rx="1.5" />
    </svg>
  );
}
function UploadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 16V4M6 10l6-6 6 6" />
      <path d="M4 20h16" />
    </svg>
  );
}
function MicIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
}
