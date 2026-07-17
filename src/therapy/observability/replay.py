"""Deterministic, network-free replay of restricted interaction journals.

Reconstruction delegates to :class:`JournalStore`, while rendering proves that
the canonical record and both request representations still match their exact
journal bytes. Live execution is possible only through an explicit injected
callable; this module never selects or contacts a provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from therapy.observability.interactions import JsonValue, canonical_json
from therapy.observability.journal import JournalError, JournalStore

_MAX_INTERACTION_ID_CHARS = 256


class ReplayError(RuntimeError):
    """Base error for deterministic replay failures."""


class ReplayNotFoundError(ReplayError):
    """Raised when a journal or interaction does not exist."""


class ReplayVerificationError(ReplayError):
    """Raised when corrupted or non-canonical evidence cannot be replayed."""


@dataclass(frozen=True, slots=True)
class ReplayEnvelope:
    """Reconstructed attempt plus its exact persisted render evidence."""

    interaction_id: str
    record: dict[str, JsonValue]
    canonical_record: dict[str, JsonValue]
    stored_canonical_json: str
    stored_canonical_request_json: str
    stored_provider_request_json: str
    checksum_verified: bool


@dataclass(frozen=True, slots=True)
class RenderedReplay:
    """Pure render of the canonical and provider-native request boundary."""

    interaction_id: str
    canonical_request: dict[str, JsonValue]
    provider_request: dict[str, JsonValue]
    canonical_json: str
    canonical_request_json: str
    provider_request_json: str
    checksum_verified: bool
    exact_match: bool

    @property
    def verified(self) -> bool:
        """Whether checksums and every byte-exact render comparison pass."""
        return self.checksum_verified and self.exact_match


def _json_object(value: object, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReplayVerificationError(f"{label} must be a JSON object")
    return cast(dict[str, JsonValue], value)


def _required_text(payload: dict[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ReplayVerificationError(f"journal field {key!r} must be non-empty text")
    return value


def _parse_object(text: str, label: str) -> dict[str, JsonValue]:
    try:
        value: object = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ReplayVerificationError(f"{label} is not valid JSON") from error
    return _json_object(value, label)


def reconstruct(journal_path: str | Path, interaction_id: str) -> ReplayEnvelope:
    """Reconstruct one complete canonical envelope from a journal.

    Args:
        journal_path: Existing restricted journal database.
        interaction_id: Exact persisted interaction identifier.

    Returns:
        Reconstructed terminal envelope and the stored request render evidence.

    Raises:
        ReplayNotFoundError: If the database or interaction does not exist.
        ReplayVerificationError: If persisted evidence cannot be reconstructed.
        ValueError: If ``interaction_id`` is invalid.
    """
    if not interaction_id or len(interaction_id) > _MAX_INTERACTION_ID_CHARS:
        raise ValueError(
            f"interaction_id must contain 1-{_MAX_INTERACTION_ID_CHARS} characters"
        )
    if "\x00" in interaction_id:
        raise ValueError("interaction_id must not contain NUL")

    path = Path(journal_path).expanduser()
    if not path.is_file():
        raise ReplayNotFoundError(f"journal does not exist: {path}")

    store = JournalStore(path)
    try:
        loaded = store.load(interaction_id)
        if loaded is None:
            raise ReplayNotFoundError(f"interaction not found: {interaction_id}")
        try:
            checksum_verified = store.verify_checksums(interaction_id)
            record = store.reconstruct(interaction_id)
        except (JournalError, TypeError, ValueError) as error:
            raise ReplayVerificationError(
                f"interaction {interaction_id!r} cannot be reconstructed"
            ) from error
    finally:
        store.close()

    if record is None:  # guarded by load(), but preserve the public invariant
        raise ReplayNotFoundError(f"interaction not found: {interaction_id}")
    row = _json_object(loaded.get("interaction"), "journal interaction row")
    stored_canonical_json = _required_text(row, "canonical_record_json")
    return ReplayEnvelope(
        interaction_id=interaction_id,
        record=record,
        canonical_record=_parse_object(stored_canonical_json, "canonical record"),
        stored_canonical_json=stored_canonical_json,
        stored_canonical_request_json=_required_text(row, "canonical_request_json"),
        stored_provider_request_json=_required_text(row, "provider_request_json"),
        checksum_verified=checksum_verified,
    )


def replay_render(envelope: ReplayEnvelope) -> RenderedReplay:
    """Purely rebuild both request representations and verify exact bytes.

    Args:
        envelope: Evidence returned by :func:`reconstruct`.

    Returns:
        Canonical JSON strings, parsed requests, and verification status.

    Raises:
        ReplayVerificationError: If required envelope objects are absent.
    """
    canonical_request = _json_object(envelope.record.get("request"), "request")
    provider_native = _json_object(
        envelope.record.get("provider_native"), "provider_native"
    )
    provider_request = _json_object(
        provider_native.get("request"), "provider_native.request"
    )
    base_request = _json_object(
        envelope.canonical_record.get("request"), "canonical record request"
    )
    base_native = _json_object(
        envelope.canonical_record.get("provider_native"),
        "canonical record provider_native",
    )
    base_provider_request = _json_object(
        base_native.get("request"), "canonical record provider_native.request"
    )

    rendered_canonical = canonical_json(envelope.canonical_record)
    rendered_request = canonical_json(canonical_request)
    rendered_provider_request = canonical_json(provider_request)
    identity_matches = (
        envelope.record.get("interaction_id") == envelope.interaction_id
        and envelope.canonical_record.get("interaction_id") == envelope.interaction_id
    )
    exact_match = (
        identity_matches
        and canonical_request == base_request
        and provider_request == base_provider_request
        and rendered_canonical == envelope.stored_canonical_json
        and rendered_request == envelope.stored_canonical_request_json
        and rendered_provider_request == envelope.stored_provider_request_json
    )
    return RenderedReplay(
        interaction_id=envelope.interaction_id,
        canonical_request=canonical_request,
        provider_request=provider_request,
        canonical_json=rendered_canonical,
        canonical_request_json=rendered_request,
        provider_request_json=rendered_provider_request,
        checksum_verified=envelope.checksum_verified,
        exact_match=exact_match,
    )


def replay_execute[T](
    envelope: ReplayEnvelope,
    executor: Callable[[dict[str, JsonValue]], T],
) -> T:
    """Execute a verified replay through an explicitly injected callable.

    Args:
        envelope: Evidence returned by :func:`reconstruct`.
        executor: Caller-owned function accepting the exact provider request.

    Returns:
        The injected executor's result.

    Raises:
        ReplayVerificationError: If checksum or exact-render verification fails.
        TypeError: If ``executor`` is not callable.
    """
    if not callable(executor):
        raise TypeError("executor must be callable")
    rendered = replay_render(envelope)
    if not rendered.verified:
        raise ReplayVerificationError(
            f"interaction {envelope.interaction_id!r} failed replay verification"
        )
    request = _parse_object(rendered.provider_request_json, "provider request")
    return executor(request)


__all__ = [
    "RenderedReplay",
    "ReplayEnvelope",
    "ReplayError",
    "ReplayNotFoundError",
    "ReplayVerificationError",
    "reconstruct",
    "replay_execute",
    "replay_render",
]
