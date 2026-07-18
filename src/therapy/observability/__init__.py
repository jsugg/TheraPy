"""Owned observability boundary (two-plane design; .local/obs-needs-impl-plan.md).

Only the small, framework-free public surface is exported here. Vendor SDKs
(OTel, backends) are confined to `telemetry.py` and `exporters.py`; Pipecat
types are confined to `therapy.integrations.pipecat.observability`.
"""

from therapy.observability.model import (
    CaptureMode,
    Component,
    Destination,
    FieldClassification,
    InteractionEventKind,
    InteractionOperation,
    InteractionStatus,
    LanguageGroup,
    Modality,
    Outcome,
    Provider,
    TelemetryPlane,
    WorkloadClass,
)

__all__ = [
    "CaptureMode",
    "Component",
    "Destination",
    "FieldClassification",
    "InteractionEventKind",
    "InteractionOperation",
    "InteractionStatus",
    "LanguageGroup",
    "Modality",
    "Outcome",
    "Provider",
    "TelemetryPlane",
    "WorkloadClass",
]
