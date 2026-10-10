import Link from "next/link";
import { WarningCircle } from "@phosphor-icons/react/dist/ssr";

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
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-surface-2 text-warn ring-1 ring-line">
        <WarningCircle size={28} aria-hidden="true" />
      </div>
      <h1 className="font-display mt-5 text-xl font-semibold text-ink">{title}</h1>
      <p className="mt-2 text-sm text-ink-soft">{message}</p>
      <div className="mt-6 flex flex-wrap justify-center gap-3">
        {action}
        <Link
          href="/"
          className="btn btn-secondary"
        >
          Về thư viện
        </Link>
      </div>
    </div>
  );
}
