"""Tests for priority and loss policy of shared model inference."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from meetasr.api.routes.health import inference_metrics
from meetasr.services.inference_coordinator import (
    InferenceCoordinator,
    InferenceKind,
    PartialInferenceDropped,
)


@pytest.mark.asyncio
async def test_live_work_overtakes_offline_work_and_calls_stay_sequential() -> None:
    order: list[str] = []
    active = 0
    max_active = 0
    started = asyncio.Event()
    release = asyncio.Event()

    def run(name: str) -> str:
        order.append(name)
        return name

    async def runner(function, *args, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        result = function(*args, **kwargs)
        if result == "running":
            started.set()
            await release.wait()
        active -= 1
        return result

    coordinator = InferenceCoordinator(runner=runner)
    await coordinator.start()

    running = asyncio.create_task(coordinator.submit(InferenceKind.UPLOAD, run, "running"))
    await started.wait()
    partial = asyncio.create_task(
        coordinator.submit(
            InferenceKind.PARTIAL,
            run,
            "partial",
            partial_key="session-1",
        )
    )
    confirmed = asyncio.create_task(coordinator.submit(InferenceKind.CONFIRMED, run, "confirmed"))
    fallback = asyncio.create_task(coordinator.submit(InferenceKind.FALLBACK, run, "fallback"))
    targeted = asyncio.create_task(coordinator.submit(InferenceKind.TARGETED, run, "targeted"))
    upload = asyncio.create_task(coordinator.submit(InferenceKind.UPLOAD, run, "upload"))
    finalize = asyncio.create_task(coordinator.submit(InferenceKind.FINALIZE, run, "finalize"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert coordinator.queue_depth == 6
    release.set()

    assert await asyncio.gather(
        running,
        confirmed,
        partial,
        fallback,
        targeted,
        upload,
        finalize,
    ) == [
        "running",
        "confirmed",
        "partial",
        "fallback",
        "targeted",
        "upload",
        "finalize",
    ]
    assert order == [
        "running",
        "confirmed",
        "partial",
        "fallback",
        "targeted",
        "upload",
        "finalize",
    ]
    assert max_active == 1
    metrics = coordinator.snapshot()
    assert metrics.asr_call_count == 6
    assert metrics.max_queue_depth >= 6
    assert metrics.completed_by_kind[InferenceKind.CONFIRMED] == 1
    assert metrics.average_wait_ms_by_kind[InferenceKind.PARTIAL] >= 0
    assert metrics.average_run_ms_by_kind[InferenceKind.PARTIAL] >= 0
    await coordinator.stop()


@pytest.mark.asyncio
async def test_only_latest_pending_partial_is_executed() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    executed: list[str] = []

    def blocker() -> None:
        return None

    def partial(name: str) -> str:
        executed.append(name)
        return name

    async def runner(function, *args, **kwargs):
        result = function(*args, **kwargs)
        if function is blocker:
            started.set()
            await release.wait()
        return result

    coordinator = InferenceCoordinator(runner=runner)
    await coordinator.start()

    running = asyncio.create_task(coordinator.submit(InferenceKind.UPLOAD, blocker))
    await started.wait()
    stale = asyncio.create_task(
        coordinator.submit(
            InferenceKind.PARTIAL,
            partial,
            "stale",
            partial_key="session-1",
        )
    )
    latest = asyncio.create_task(
        coordinator.submit(
            InferenceKind.PARTIAL,
            partial,
            "latest",
            partial_key="session-1",
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert coordinator.queue_depth == 2
    release.set()

    await running
    with pytest.raises(PartialInferenceDropped):
        await stale
    assert await latest == "latest"
    assert executed == ["latest"]
    metrics = coordinator.snapshot()
    assert metrics.partial_drop_count == 1
    assert metrics.asr_call_count == 2
    await coordinator.stop()


@pytest.mark.asyncio
async def test_fallback_is_never_dropped_and_is_counted() -> None:
    async def runner(function, *args, **kwargs):
        return function(*args, **kwargs)

    coordinator = InferenceCoordinator(runner=runner)
    await coordinator.start()

    result = await coordinator.submit(
        InferenceKind.FALLBACK,
        lambda: "recovered",
    )

    assert result == "recovered"
    metrics = coordinator.snapshot()
    assert metrics.fallback_count == 1
    assert metrics.completed_by_kind[InferenceKind.FALLBACK] == 1
    assert metrics.average_wait_ms >= 0
    assert metrics.average_run_ms >= 0
    await coordinator.stop()


@pytest.mark.asyncio
async def test_targeted_asr_is_counted_without_incrementing_full_fallback() -> None:
    async def runner(function, *args, **kwargs):
        return function(*args, **kwargs)

    coordinator = InferenceCoordinator(runner=runner)
    await coordinator.start()

    result = await coordinator.submit(
        InferenceKind.TARGETED,
        lambda: "targeted",
    )

    assert result == "targeted"
    metrics = coordinator.snapshot()
    assert metrics.asr_call_count == 1
    assert metrics.fallback_count == 0
    assert metrics.completed_by_kind[InferenceKind.TARGETED] == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_live_snapshot() -> None:
    async def runner(function, *args, **kwargs):
        return function(*args, **kwargs)

    coordinator = InferenceCoordinator(runner=runner)
    await coordinator.start()
    await coordinator.submit(InferenceKind.CONFIRMED, lambda: "ok")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(inference_coordinator=coordinator))
    )

    payload = await inference_metrics(request)

    assert payload["available"] is True
    assert payload["queue_depth"] == 0
    assert payload["asr_call_count"] == 1
    assert payload["completed_by_kind"] == {"confirmed": 1}
    assert payload["average_wait_ms_by_kind"]["confirmed"] >= 0
    assert payload["average_run_ms_by_kind"]["confirmed"] >= 0
    await coordinator.stop()
