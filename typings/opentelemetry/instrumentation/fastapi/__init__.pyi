from collections.abc import Callable

from fastapi import FastAPI
from opentelemetry.trace import Span, TracerProvider

type ServerRequestHook = Callable[[Span, dict[str, object]], None]


class FastAPIInstrumentor:
    @staticmethod
    def instrument_app(
        app: FastAPI,
        *,
        excluded_urls: str | None = ...,
        tracer_provider: TracerProvider | None = ...,
        server_request_hook: ServerRequestHook | None = ...,
        exclude_spans: list[str] | None = ...,
    ) -> None: ...

    @staticmethod
    def uninstrument_app(app: FastAPI) -> None: ...
