import Link from "next/link";

/** Full-page friendly message used by error.tsx / not-found.tsx. */
export function ErrorPage({
  title,
  message,
  action,
}: {
  title: string;
  message: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="mx-auto flex min-h-[70vh] max-w-md flex-col items-center justify-center px-6 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-signal-wash text-warn">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12.5" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      </div>
      <h1 className="font-display mt-5 text-xl font-semibold text-ink">{title}</h1>
      <p className="mt-2 text-sm text-ink-soft">{message}</p>
      <div className="mt-6 flex flex-wrap justify-center gap-3">
        {action}
        <Link
          href="/"
          className="rounded-xl border border-line px-4 py-2.5 text-sm font-medium text-ink-soft transition hover:border-brand hover:text-brand"
        >
          Về thư viện
        </Link>
      </div>
    </div>
  );
}
