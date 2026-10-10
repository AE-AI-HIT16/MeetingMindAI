/** Library skeleton: same shape as the stat tiles and source grid. */
export default function LibraryLoading() {
  return (
    <div
      className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 md:px-10 md:py-10"
      aria-busy="true"
      aria-label="Đang tải thư viện"
    >
      <div className="skeleton h-10 w-64 rounded-lg" />
      <div className="mt-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="skeleton h-[88px] rounded-xl" />
        ))}
      </div>
      <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }, (_, i) => (
          <div key={i} className="skeleton h-[260px] rounded-[var(--radius-card)]" />
        ))}
      </div>
    </div>
  );
}
