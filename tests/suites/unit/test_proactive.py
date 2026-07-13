"""Tests for the proactivity engine: quiet hours + never_initiate (W5)."""

from __future__ import annotations

from datetime import datetime

from therapy.dialogue import proactive
from therapy.dialogue.proactive import (
    CHECK_IN,
    ChannelConfig,
    ProactivityConfig,
    QuietHours,
    should_fire,
)


def _at(hour: int) -> datetime:
    return datetime(2026, 7, 12, hour, 0, 0)


def _check_in_config(quiet: QuietHours) -> ProactivityConfig:
    return ProactivityConfig(
        channels={CHECK_IN: ChannelConfig(enabled=True, quiet_hours=quiet)}
    )


def test_quiet_hours_overnight_window_wraps_midnight() -> None:
    quiet = QuietHours(start=22, end=8)
    assert quiet.contains(_at(23)) is True
    assert quiet.contains(_at(3)) is True
    assert quiet.contains(_at(12)) is False


def test_channel_fires_outside_quiet_hours_and_is_suppressed_within() -> None:
    config = _check_in_config(QuietHours(start=22, end=8))
    # Within quiet hours: suppressed. Outside: fires.
    assert should_fire(CHECK_IN, _at(2), config) is False
    assert should_fire(CHECK_IN, _at(14), config) is True


def test_disabled_and_unknown_channels_never_fire() -> None:
    config = _check_in_config(QuietHours(start=22, end=8))
    assert should_fire("push", _at(14), config) is False  # not enabled/configured
    assert should_fire("not_a_channel", _at(14), config) is False


def test_never_initiate_topic_blocks_outreach() -> None:
    config = _check_in_config(QuietHours(start=22, end=8))
    assert (
        should_fire(
            CHECK_IN,
            _at(14),
            config,
            topic="how are things with your father",
            never_initiate=["father"],
        )
        is False
    )
    assert (
        should_fire(
            CHECK_IN, _at(14), config, topic="how did the run go", never_initiate=["father"]
        )
        is True
    )


def test_due_channels_lists_only_cleared_channels() -> None:
    config = ProactivityConfig(
        channels={
            CHECK_IN: ChannelConfig(enabled=True, quiet_hours=QuietHours(22, 8)),
            proactive.DIGEST: ChannelConfig(enabled=False),
        }
    )
    assert proactive.due_channels(_at(14), config) == [CHECK_IN]
    assert proactive.due_channels(_at(2), config) == []
