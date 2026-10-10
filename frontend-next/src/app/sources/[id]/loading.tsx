/** Source / document skeleton: title, tabs, player, then text lines. */
export default function SourceLoading() {
  return (
    <div
      className="mx-auto w-full max-w-4xl px-4 py-8 sm:px-6 md:px-10 md:py-10"
      aria-busy="true"
      aria-label="Đang tải tài liệu"
    >
      <div className="skeleton h-3 w-20 rounded" />
      <div className="skeleton mt-4 h-9 w-3/4 rounded-lg" />
      <div className="skeleton mt-3 h-4 w-48 rounded" />
      <div className="skeleton mt-8 h-10 w-full rounded-lg" />
      <div className="skeleton mt-4 h-20 w-full rounded-2xl" />
      <div className="mt-8 space-y-3">
        {["w-full", "w-11/12", "w-full", "w-4/5", "w-full", "w-2/3"].map((w, i) => (
          <div key={i} className={`skeleton h-4 rounded ${w}`} />
        ))}
      </div>
    </div>
  );
}
