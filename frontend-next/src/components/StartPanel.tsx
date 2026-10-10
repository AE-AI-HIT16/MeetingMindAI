import Link from "next/link";
import { ArrowRight, Microphone, UploadSimple } from "@phosphor-icons/react/dist/ssr";
import { Waveform } from "@/components/ui";

/** The two ways to start a document: record live, or upload a file. */
export function StartPanel({ large = false }: { large?: boolean }) {
  const height = large ? "min-h-[260px]" : "min-h-[180px]";
  const title = `font-display font-semibold tracking-tight text-ink ${large ? "text-2xl" : "text-xl"}`;

  return (
    <div className="grid gap-4 md:grid-cols-5">
      <Link
        href="/realtime"
        className="bezel group block transition duration-500 hover:-translate-y-1 md:col-span-3"
      >
        <div className={`bezel-core relative flex h-full flex-col justify-between overflow-hidden p-7 ${height}`}>
          {/* Warm pool of light behind the live waveform */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full bg-brand/15 opacity-70 blur-3xl transition-opacity duration-500 group-hover:opacity-100"
          />
          <div className="relative flex items-start justify-between gap-4">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand text-on-brand shadow-[inset_0_1px_0_rgb(255_255_255/0.25)]">
              <Microphone size={22} weight="fill" />
            </span>
            <Waveform live bars={large ? 28 : 20} className="h-10 w-40 opacity-80" />
          </div>
          <div className="relative mt-8">
            <h2 className={title}>Ghi âm trực tiếp</h2>
            <p className="mt-1 max-w-sm text-sm text-ink-soft">
              Lời thoại hiện ngay khi bạn nói. Dùng micro hoặc âm thanh từ một tab.
            </p>
            <span className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-ink">
              Bắt đầu ghi
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-ink text-surface transition duration-300 group-hover:-translate-y-px group-hover:translate-x-0.5">
                <ArrowRight size={14} weight="bold" />
              </span>
            </span>
          </div>
        </div>
      </Link>

      <Link
        href="/upload"
        className="bezel group block transition duration-500 hover:-translate-y-1 md:col-span-2"
      >
        <div className={`bezel-core flex h-full flex-col justify-between p-7 ${height}`}>
          <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-surface-2 text-ink ring-1 ring-line">
            <UploadSimple size={22} />
          </span>
          <div className="mt-8">
            <h2 className={title}>Tải file lên</h2>
            <p className="mt-1 text-sm text-ink-soft">Video hoặc audio, tối đa 2 GB.</p>
            <span className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-ink">
              Chọn file
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-surface-3 text-ink transition duration-300 group-hover:-translate-y-px group-hover:translate-x-0.5">
                <ArrowRight size={14} weight="bold" />
              </span>
            </span>
          </div>
        </div>
      </Link>
    </div>
  );
}
