"""Strict structural contracts for partially typed test dependencies."""

from collections.abc import Callable, MutableMapping
from typing import Protocol

import httpx


class HttpTestClient(Protocol):
    """HTTP surface used by FastAPI integration tests."""

    headers: MutableMapping[str, str]

    def request(self, method: str, url: str, **kwargs: object) -> httpx.Response: ...

    def get(self, url: str, **kwargs: object) -> httpx.Response: ...

    def post(self, url: str, **kwargs: object) -> httpx.Response: ...

    def put(self, url: str, **kwargs: object) -> httpx.Response: ...

    def patch(self, url: str, **kwargs: object) -> httpx.Response: ...

    def delete(self, url: str, **kwargs: object) -> httpx.Response: ...


type FreePort = Callable[[], int]
type MetricCall = tuple[str, float, dict[str, str]]
type EventCall = tuple[str, dict[str, object]]


class MetricRecorder(Protocol):
    """Callable metric sink used by observability tests."""

    def __call__(
        self,
        name: str,
        value: float,
        attributes: dict[str, str] | None = None,
    ) -> None: ...


class EventRecorder(Protocol):
    """Callable event sink used by structured-event tests."""

    def __call__(self, event_name: str, **kwargs: object) -> None: ...


def metric_recorder(calls: list[MetricCall]) -> MetricRecorder:
    """Create a typed metric sink that appends calls to ``calls``."""

    def record(
        name: str,
        value: float,
        attributes: dict[str, str] | None = None,
    ) -> None:
        calls.append((name, value, attributes or {}))

    return record


def event_recorder(calls: list[EventCall]) -> EventRecorder:
    """Create a typed event sink that appends calls to ``calls``."""

    def record(event_name: str, **kwargs: object) -> None:
        calls.append((event_name, kwargs))

    return record


class WaitUntil(Protocol):
    """Polling fixture contract."""

    def __call__(
        self, predicate: Callable[[], bool], timeout: float = 3.0
    ) -> None: ...
