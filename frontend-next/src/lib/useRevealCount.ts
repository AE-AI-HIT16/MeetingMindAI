"use client";

import { useEffect, useState } from "react";

/**
 * Delay before revealing the next segment: offline ASR returns segments in
 * batches, so show them one by one (faster when many are waiting) instead of
 * a burst every few seconds that makes the page look frozen in between.
 */
export function revealDelayMs(backlog: number): number {
  if (backlog > 12) return 60;
  if (backlog > 4) return 150;
  return 300;
}

/** How many of `total` items to show; the first `immediate` show at once. */
export function useRevealCount(total: number, immediate: number): number {
  const [shown, setShown] = useState(0);
  const visible = Math.min(total, Math.max(shown, immediate));

  useEffect(() => {
    if (visible >= total) return;
    const timer = setTimeout(() => setShown(visible + 1), revealDelayMs(total - visible));
    return () => clearTimeout(timer);
  }, [visible, total]);

  return visible;
}
