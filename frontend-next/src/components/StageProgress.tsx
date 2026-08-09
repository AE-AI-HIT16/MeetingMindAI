import type { ProcessingStage } from "@/lib/types";

const STAGES: { key: ProcessingStage; label: string }[] = [
  { key: "extracting_audio", label: "Tách audio" },
  { key: "transcribing", label: "Nhận dạng lời nói" },
  { key: "generating_doc", label: "Tạo tài liệu" },
];

export function StageProgress({
  current,
  progress,
}: {
  current: ProcessingStage;
  progress: number; // 0..1 within current stage
}) {
  const idx = STAGES.findIndex((s) => s.key === current);
  return (
    <div className="flex items-center gap-2">
      {STAGES.map((s, i) => {
        const done = i < idx;
        const active = i === idx;
        return (
          <div key={s.key} className="flex flex-1 items-center gap-2">
            <div className="flex-1">
              <div className="flex items-center gap-1.5">
                <span
                  className={`font-mono text-[11px] font-bold ${
                    active
                      ? "text-signal-ink"
                      : done
                        ? "text-ok"
                        : "text-ink-faint"
                  }`}
                >
                  0{i + 1}
                </span>
                <span
                  className={`text-xs font-medium ${
                    active || done ? "text-ink" : "text-ink-faint"
                  }`}
                >
                  {s.label}
                </span>
              </div>
              <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-line">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    done ? "bg-ok" : "bg-signal"
                  }`}
                  style={{
                    width: done ? "100%" : active ? `${progress * 100}%` : "0%",
                  }}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
