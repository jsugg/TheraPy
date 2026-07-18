"""Component health snapshots and the readiness model (plan §3, O3.1).

Framework-free registry. `/ready` (O3.1) renders `snapshot()` as enums only
— no paths, no errors, no IDs. Liveness (`/health`) never consults this.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import StrEnum

from therapy.observability.model import Component


class ComponentState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    component: Component
    state: ComponentState
    #: Unix timestamp of the last state change; ages are computed by readers.
    changed_at: float
    #: Bounded reason enum-ish token (never free text / exception payload).
    reason: str = "none"


class HealthRegistry:
    """Thread-safe component state map; the single readiness source."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._components: dict[Component, ComponentHealth] = {}

    def set_state(
        self, component: Component, state: ComponentState, reason: str = "none"
    ) -> None:
        with self._lock:
            current = self._components.get(component)
            if current is not None and current.state is state:
                return
            self._components[component] = ComponentHealth(
                component=component,
                state=state,
                changed_at=time.time(),
                reason=reason,
            )

    def snapshot(self) -> dict[str, dict[str, object]]:
        with self._lock:
            return {
                health.component.value: {
                    "state": health.state.value,
                    "changed_at_unixtime": int(health.changed_at),
                    "reason": health.reason,
                }
                for health in self._components.values()
            }

    def degraded_components(self) -> list[Component]:
        with self._lock:
            return [
                health.component
                for health in self._components.values()
                if health.state is ComponentState.DEGRADED
            ]


_registry = HealthRegistry()


def registry() -> HealthRegistry:
    return _registry
