"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import { useState, useEffect } from "react";
import {
  Books,
  Microphone,
  SignIn,
  SignOut,
  UploadSimple,
  type Icon,
} from "@phosphor-icons/react";
import { getStorageSummary } from "@/lib/api";
import type { StorageSummary } from "@/lib/types";

const NAV: { href: string; label: string; icon: Icon }[] = [
  { href: "/", label: "Thư viện", icon: Books },
  { href: "/upload", label: "Tải lên", icon: UploadSimple },
  { href: "/realtime", label: "Ghi âm trực tiếp", icon: Microphone },
];

function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

export function Sidebar() {
  const pathname = usePathname();
  // The login screen is a standalone page.
  if (pathname.startsWith("/login")) return null;

  return (
    <aside className="sticky top-3 m-3 mr-0 hidden h-[calc(100dvh-1.5rem)] w-64 shrink-0 flex-col rounded-[var(--radius-bezel)] bg-surface/80 px-4 py-6 shadow-[var(--shadow-card)] ring-1 ring-line/70 backdrop-blur-xl md:flex">
      <Link href="/" className="mb-10 flex items-center gap-2.5 px-2">
        <Logo />
        <span className="font-display text-[19px] font-semibold tracking-tight text-ink">
          MeetingMind
        </span>
      </Link>

      <nav aria-label="Điều hướng chính" className="flex flex-col gap-0.5">
        {NAV.map(({ href, label, icon: NavIcon }) => {
          const active = isActive(pathname, href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={`relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${
                active
                  ? "bg-surface-3/70 text-ink"
                  : "text-ink-soft hover:bg-surface-2 hover:text-ink"
              }`}
            >
              {active && (
                <span
                  aria-hidden="true"
                  className="absolute inset-y-2 left-0 w-[3px] rounded-r-full bg-brand"
                />
              )}
              <NavIcon
                size={18}
                weight={active ? "fill" : "regular"}
                className={active ? "text-brand" : undefined}
              />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Account and storage sit at the bottom, out of the way of navigation */}
      <div className="mt-auto space-y-3">
        <UserSection />
        <StorageWidget />
      </div>
    </aside>
  );
}

/** Below md the sidebar is hidden; this bar keeps navigation reachable. */
export function MobileNav() {
  const pathname = usePathname();
  const { status } = useSession();
  if (pathname.startsWith("/login")) return null;

  return (
    <header className="border-b border-line bg-surface/80 backdrop-blur md:hidden">
      <div className="flex items-center justify-between px-4 pt-3">
        <Link href="/" className="flex items-center gap-2">
          <Logo />
          <span className="font-display text-[17px] font-semibold tracking-tight text-ink">
            MeetingMind
          </span>
        </Link>
        {status === "authenticated" ? (
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-medium text-ink-soft transition hover:text-danger-ink"
          >
            <SignOut size={14} />
            Đăng xuất
          </button>
        ) : status === "unauthenticated" ? (
          <Link
            href="/login"
            className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-medium text-brand-ink transition hover:bg-brand-wash"
          >
            <SignIn size={14} />
            Đăng nhập
          </Link>
        ) : null}
      </div>
      <nav aria-label="Điều hướng chính" className="mt-2 flex gap-1 overflow-x-auto px-3 pb-2">
        {NAV.map(({ href, label, icon: NavIcon }) => {
          const active = isActive(pathname, href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={`flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition ${
                active
                  ? "bg-surface-2 text-ink"
                  : "text-ink-soft hover:bg-surface-2/60 hover:text-ink"
              }`}
            >
              <NavIcon
                size={17}
                weight={active ? "fill" : "regular"}
                className={active ? "text-brand" : undefined}
              />
              {label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function StorageWidget() {
  const { data: session, status } = useSession();
  const [storage, setStorage] = useState<StorageSummary | null>(null);

  useEffect(() => {
    if (!session?.accessToken) return;
    getStorageSummary(session.accessToken).then(setStorage).catch(() => {});
  }, [session?.accessToken]);

  const used = storage?.used_bytes ?? 0;
  const quota = storage?.quota_bytes ?? 20 * 1024 * 1024 * 1024;
  const pct = quota > 0 ? Math.min((used / quota) * 100, 100) : 0;

  // Guests store nothing, so a quota would only be noise.
  if (status !== "authenticated") return null;

  return (
    <div className="rounded-xl border border-line bg-surface-2/60 p-4">
      <div className="flex items-baseline justify-between gap-2">
        <p className="eyebrow">Dung lượng</p>
        <p className="font-mono text-[11px] tabular-nums text-ink-faint">
          {pct.toFixed(1)}%
        </p>
      </div>
      <p className="mt-1.5 font-mono text-xs tabular-nums text-ink-soft">
        {formatBytes(used)} / {formatBytes(quota)}
      </p>
      <div className="mt-2.5 h-1 overflow-hidden rounded-full bg-line">
        <div className="h-full rounded-full bg-brand transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function UserSection() {
  const { data: session, status } = useSession();

  if (status === "loading") return null;

  if (status === "authenticated" && session?.user) {
    return (
      <div className="flex items-center gap-2.5 rounded-xl border border-line bg-surface-2/60 p-3">
        {session.user.image ? (
          <Image
            src={session.user.image}
            alt={session.user.name ?? "Ảnh đại diện"}
            width={32}
            height={32}
            unoptimized
            className="shrink-0 rounded-lg ring-1 ring-line"
          />
        ) : (
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand text-xs font-semibold text-on-brand">
            {session.user.name?.[0]?.toUpperCase() ?? "U"}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium leading-tight text-ink">
            {session.user.name}
          </p>
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className="text-xs text-ink-faint transition hover:text-danger-ink"
          >
            Đăng xuất
          </button>
        </div>
      </div>
    );
  }

  return (
    <Link
      href="/login"
      className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-dashed border-line px-4 py-2.5 text-xs font-medium text-ink-soft transition hover:border-brand hover:text-brand-ink"
    >
      <SignIn size={14} />
      Đăng nhập để lưu dữ liệu
    </Link>
  );
}

export function Logo({ size = 32 }: { size?: number }) {
  // A sound mark resolving into a baseline: voice becoming text.
  return (
    <span
      className="flex shrink-0 items-center justify-center rounded-lg bg-surface-3 ring-1 ring-line"
      style={{ width: size, height: size }}
    >
      <svg width={size * 0.56} height={size * 0.56} viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <rect x="1" y="7" width="2" height="4" rx="1" fill="var(--color-ink)" />
        <rect x="5" y="3" width="2" height="12" rx="1" fill="var(--color-ink)" />
        <rect x="9" y="5" width="2" height="8" rx="1" fill="var(--color-ink)" />
        <rect x="13" y="8" width="2" height="2" rx="1" fill="var(--color-brand)" />
      </svg>
    </span>
  );
}
