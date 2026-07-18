"""Proactivity engine: assistant-initiated contact within user boundaries.

Framework-free (SPEC dependency boundary). Four channels (SPEC §3), each
individually configurable and each honoring quiet hours: PWA **push**, in-app
**greeting**, scheduled **check-in**, and a written **digest**. Every outreach
consults the user model's `never_initiate` boundaries first.

This serves reflection, never engagement: there are deliberately no streaks,
no guilt nudges, no "you haven't talked to me in N days" mechanics (SPEC §4).
The scheduler decides *whether* a channel may fire right now; composing the
actual message is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# The four channels (SPEC §3). Kept as plain strings so config/JSON round-trips
# without an enum import at every call site.
PUSH = "push"
GREETING = "greeting"
CHECK_IN = "check_in"
DIGEST = "digest"
CHANNELS: tuple[str, ...] = (PUSH, GREETING, CHECK_IN, DIGEST)


@dataclass(frozen=True)
class QuietHours:
    """A daily do-not-disturb window in local wall-clock hours [0, 24).

    `start == end` means no quiet window (always allowed). The window wraps
    past midnight when `start > end` (e.g. 22->8 is overnight), so quiet hours
    that straddle midnight are handled without special-casing at call sites.
    """

    start: int = 22
    end: int = 8

    def contains(self, when: datetime) -> bool:
        """Whether `when` falls inside the quiet window."""
        if self.start == self.end:
            return False
        hour = when.hour
        if self.start < self.end:
            return self.start <= hour < self.end
        return hour >= self.start or hour < self.end


@dataclass
class ChannelConfig:
    """Per-channel proactivity settings: on/off plus its own quiet hours."""

    enabled: bool = False
    quiet_hours: QuietHours = field(default_factory=QuietHours)


@dataclass
class ProactivityConfig:
    """The user's proactivity preferences across all four channels.

    Every channel defaults to *off*: the assistant reaches out only where the
    user opted in. `for_channel` returns a disabled default for any channel not
    explicitly configured, so an unknown channel can never fire.
    """

    channels: dict[str, ChannelConfig] = field(default_factory=lambda: {})

    def for_channel(self, channel: str) -> ChannelConfig:
        """Return a channel's config, or a disabled default."""
        return self.channels.get(channel, ChannelConfig(enabled=False))


def within_quiet_hours(
    channel: str, config: ProactivityConfig, when: datetime
) -> bool:
    """Whether `when` is inside the channel's quiet window."""
    return config.for_channel(channel).quiet_hours.contains(when)


def should_fire(
    channel: str,
    when: datetime,
    config: ProactivityConfig,
    *,
    topic: str | None = None,
    never_initiate: list[str] | None = None,
) -> bool:
    """Whether `channel` may reach out at `when` — the single outreach gate.

    Fires only when the channel is enabled, `when` is *outside* its quiet
    hours, and — when a `topic` is given — that topic is not on the
    `never_initiate` list. Any unknown channel is refused.
    """
    if channel not in CHANNELS:
        return False
    channel_config = config.for_channel(channel)
    if not channel_config.enabled:
        return False
    if channel_config.quiet_hours.contains(when):
        return False
    if topic is not None and _matches_never_initiate(topic, never_initiate or []):
        return False
    return True


def due_channels(
    when: datetime,
    config: ProactivityConfig,
    *,
    topic: str | None = None,
    never_initiate: list[str] | None = None,
) -> list[str]:
    """All channels cleared to fire at `when`, in canonical order."""
    return [
        channel
        for channel in CHANNELS
        if should_fire(
            channel, when, config, topic=topic, never_initiate=never_initiate
        )
    ]


def _matches_never_initiate(topic: str, never_initiate: list[str]) -> bool:
    """Whether an outreach topic touches a never-initiate boundary."""
    lowered = topic.lower()
    return any(banned.lower() in lowered for banned in never_initiate)
