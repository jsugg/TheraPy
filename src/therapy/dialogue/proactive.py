"""Proactivity engine: assistant-initiated contact within user boundaries.

Channels (SPEC §3, each individually configurable, quiet hours honored):
push notification, in-app greeting, scheduled check-in, written digest.
Must consult the user model's `never_initiate` boundaries before any
outreach. Serves growth and reflection, never engagement. Phase 4.
"""
