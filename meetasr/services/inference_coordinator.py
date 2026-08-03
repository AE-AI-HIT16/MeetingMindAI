"""Priority scheduler for every call into the shared ASR model."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class InferenceKind(str, Enum):
    """Work classes ordered by whether their output may be discarded."""

    CONFIRMED = "confirmed"
    FALLBACK = "fallback"
    TARGETED = "targeted"
    UPLOAD = "upload"
    FINALIZE = "finalize"
    PARTIAL = "partial"


_PRIORITY = {
    InferenceKind.CONFIRMED: 0,
    InferenceKind.FALLBACK: 5,
    InferenceKind.TARGETED: 7,
    InferenceKind.UPLOAD: 10,
    InferenceKind.FINALIZE: 15,
    InferenceKind.PARTIAL: 30,
}


class PartialInferenceDropped(RuntimeError):
    """A disposable preview was replaced before model execution."""


@dataclass(frozen=True, slots=True)
class InferenceMetricsSnapshot:
    """Small in-process metrics view for health logging and tests."""

    queue_depth: int
    max_queue_depth: int
    asr_call_count: int
    fallback_count: int
    partial_drop_count: int
    failure_count: int
    average_wait_ms: float
    average_run_ms: float
    average_wait_ms_by_kind: dict[InferenceKind, float]
    average_run_ms_by_kind: dict[InferenceKind, float]
    completed_by_kind: dict[InferenceKind, int]


@dataclass(order=True, slots=True)
class _InferenceRequest:
    priority: int
    sequence: int
    kind: InferenceKind = field(compare=False)
    function: Callable[..., Any] = field(compare=False)
    args: tuple[Any, ...] = field(compare=False)
    kwargs: dict[str, Any] = field(compare=False)
    future: asyncio.Future[Any] = field(compare=False)
    enqueued_at: float = field(compare=False)
    partial_key: str | None = field(compare=False, default=None)
    partial_generation: int = field(compare=False, default=0)


class InferenceCoordinator:
    """Run shared-model work sequentially with confirmed-first priority."""

    def __init__(
        self,
        runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._queue: asyncio.PriorityQueue[_InferenceRequest] = (
            asyncio.PriorityQueue()
        )
        self._sequence = itertools.count()
        self._worker_task: asyncio.Task[None] | None = None
        self._accepting = False
        self._partial_generations: dict[str, int] = {}
        self._max_queue_depth = 0
        self._asr_call_count = 0
        self._fallback_count = 0
        self._partial_drop_count = 0
        self._failure_count = 0
        self._total_wait_ms = 0.0
        self._total_run_ms = 0.0
        self._wait_ms_by_kind: Counter[InferenceKind] = Counter()
        self._run_ms_by_kind: Counter[InferenceKind] = Counter()
        self._executed_by_kind: Counter[InferenceKind] = Counter()
        self._completed: Counter[InferenceKind] = Counter()
        self._runner = runner or asyncio.to_thread

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the sole model consumer."""
        if self._worker_task is not None:
            return
        self._accepting = True
        self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Drain lossless work and stop the consumer."""
        self._accepting = False
        if self._worker_task is None:
            return
        await self._queue.join()
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def submit(
        self,
        kind: InferenceKind,
        function: Callable[..., Any],
        *args: Any,
        partial_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Schedule one synchronous call and await its result."""
        if not self._accepting or self._worker_task is None:
            raise RuntimeError("inference coordinator is not running")
        generation = 0
        if kind is InferenceKind.PARTIAL:
            if not partial_key:
                raise ValueError("partial inference requires a stable key")
            generation = self._partial_generations.get(partial_key, 0) + 1
            self._partial_generations[partial_key] = generation

        loop = asyncio.get_running_loop()
        request = _InferenceRequest(
            priority=_PRIORITY[kind],
            sequence=next(self._sequence),
            kind=kind,
            function=function,
            args=args,
            kwargs=kwargs,
            future=loop.create_future(),
            enqueued_at=time.perf_counter(),
            partial_key=partial_key,
            partial_generation=generation,
        )
        await self._queue.put(request)
        self._max_queue_depth = max(self._max_queue_depth, self.queue_depth)
        return await request.future

    def snapshot(self) -> InferenceMetricsSnapshot:
        executed = sum(self._completed.values()) + self._failure_count
        average_wait_by_kind = {
            kind: self._wait_ms_by_kind[kind] / count
            for kind, count in self._executed_by_kind.items()
            if count
        }
        average_run_by_kind = {
            kind: self._run_ms_by_kind[kind] / count
            for kind, count in self._executed_by_kind.items()
            if count
        }
        return InferenceMetricsSnapshot(
            queue_depth=self.queue_depth,
            max_queue_depth=self._max_queue_depth,
            asr_call_count=self._asr_call_count,
            fallback_count=self._fallback_count,
            partial_drop_count=self._partial_drop_count,
            failure_count=self._failure_count,
            average_wait_ms=self._total_wait_ms / executed if executed else 0.0,
            average_run_ms=self._total_run_ms / executed if executed else 0.0,
            average_wait_ms_by_kind=average_wait_by_kind,
            average_run_ms_by_kind=average_run_by_kind,
            completed_by_kind=dict(self._completed),
        )

    async def _run(self) -> None:
        while True:
            request = await self._queue.get()
            try:
                await self._execute(request)
            finally:
                self._queue.task_done()

    async def _execute(self, request: _InferenceRequest) -> None:
        if request.future.cancelled():
            return
        if self._is_stale_partial(request):
            self._partial_drop_count += 1
            request.future.set_exception(
                PartialInferenceDropped("newer partial request is pending")
            )
            return

        wait_ms = (
            time.perf_counter() - request.enqueued_at
        ) * 1000
        self._total_wait_ms += wait_ms
        self._wait_ms_by_kind[request.kind] += wait_ms
        if request.kind is not InferenceKind.FINALIZE:
            self._asr_call_count += 1
        if request.kind is InferenceKind.FALLBACK:
            self._fallback_count += 1
        started_at = time.perf_counter()
        try:
            result = await self._runner(
                request.function,
                *request.args,
                **request.kwargs,
            )
        except Exception as exc:
            self._failure_count += 1
            if not request.future.cancelled():
                request.future.set_exception(exc)
        else:
            self._completed[request.kind] += 1
            if not request.future.cancelled():
                request.future.set_result(result)
        finally:
            run_ms = (
                time.perf_counter() - started_at
            ) * 1000
            self._total_run_ms += run_ms
            self._run_ms_by_kind[request.kind] += run_ms
            self._executed_by_kind[request.kind] += 1

    def _is_stale_partial(self, request: _InferenceRequest) -> bool:
        return (
            request.kind is InferenceKind.PARTIAL
            and request.partial_key is not None
            and self._partial_generations.get(request.partial_key)
            != request.partial_generation
        )
