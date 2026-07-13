"""SQLite session memory and local audio archive (SPEC §8).

This module is framework-free storage for the local-first privacy model:
session transcripts and user-model v1 facts stay in SQLite, while raw
utterance audio is archived under the same owner-controlled data directory.
Each public method opens its own SQLite connection, so calls from asyncio
callbacks do not share connections across threads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
import shutil
import sqlite3
import uuid
import wave

JsonScalar = str | int | None
RowDict = dict[str, JsonScalar]


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _row_dict(row: sqlite3.Row) -> RowDict:
    """Convert a SQLite row into a JSON-serializable dictionary."""
    return dict(row)


def resume_window_secs() -> float:
    """How long after its last activity a session is still resumable."""
    return float(os.environ.get("THERAPY_RESUME_WINDOW_SECS", "900"))


class MemoryStore:
    """Persistent session transcripts, summaries, facts, and audio."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Create the data directory and initialize the SQLite schema.

        Args:
            data_dir: Base directory for `therapy.db` and the audio archive. When
                omitted, `THERAPY_DATA_DIR` is used, then `./data`.
        """
        self.data_dir = data_dir or Path(os.environ.get("THERAPY_DATA_DIR", "./data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "therapy.db"
        self._audio_dir = self.data_dir / "audio"
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a configured SQLite connection for one method call."""
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    def _init_schema(self) -> None:
        """Create storage tables if this is a new database."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    summary TEXT,
                    title TEXT
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL
                        REFERENCES sessions(id) ON DELETE CASCADE,
                    ts TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    modality TEXT NOT NULL CHECK(modality IN ('voice','text')),
                    language TEXT NOT NULL,
                    text TEXT NOT NULL,
                    audio_path TEXT
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL DEFAULT 'observation',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    n_occurrences INTEGER NOT NULL DEFAULT 1
                );
                """
            )
            # Databases created before titles existed (2026-07-10) lack the
            # column; CREATE TABLE IF NOT EXISTS won't add it.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sessions)")
            }
            if "title" not in columns:
                with connection:
                    connection.execute("ALTER TABLE sessions ADD COLUMN title TEXT")

    def create_session(self) -> str:
        """Create a session row and return its UUID4 hex id."""
        session_id = uuid.uuid4().hex
        with self._connect() as connection:
            with connection:
                connection.execute(
                    "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
                    (session_id, _utc_now()),
                )
        return session_id

    def end_session(self, session_id: str, summary: str | None = None) -> None:
        """Mark a session ended and store its summary."""
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE sessions
                    SET ended_at = ?, summary = ?
                    WHERE id = ?
                    """,
                    (_utc_now(), summary, session_id),
                )

    def resume_candidate(
        self, window_secs: float, now: datetime | None = None
    ) -> str | None:
        """Return the newest session id when it is recent enough to resume.

        This backs reconnect-resume: a dropped WebRTC connection should not
        split one conversation into two sessions.

        Only sessions with at least one turn are candidates. A connectivity
        probe, a `netcheck` run, or a connect that dropped before the user
        said anything creates an empty session; offering it as resumable put
        a "Resume conversation" button in front of the user with nothing to
        resume (field test 2026-07-11).
        """
        if window_secs <= 0:
            return None
        if now is None:
            now = datetime.now(UTC)

        with self._connect() as connection:
            session = connection.execute(
                """
                SELECT id, started_at, ended_at
                FROM sessions
                WHERE EXISTS (
                    SELECT 1 FROM turns WHERE turns.session_id = sessions.id
                )
                ORDER BY started_at DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
            if session is None:
                return None

            session_id = str(session["id"])
            started_at = str(session["started_at"])
            ended_at = session["ended_at"]
            if ended_at is not None:
                activity_ts = str(ended_at)
            else:
                turn = connection.execute(
                    """
                    SELECT ts
                    FROM turns
                    WHERE session_id = ?
                    ORDER BY ts DESC, id DESC
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                activity_ts = str(turn["ts"]) if turn is not None else started_at

        last_activity = datetime.fromisoformat(activity_ts)
        if (now - last_activity).total_seconds() <= window_secs:
            return session_id
        return None

    def has_session(self, session_id: str) -> bool:
        """Whether a session row exists (guards explicit-resume requests)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def delete_session(self, session_id: str) -> bool:
        """Delete one session, its turns (FK cascade), and its audio archive.

        Returns:
            True when a session row was deleted, False for an unknown id.
        """
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM sessions WHERE id = ?", (session_id,)
                )
        shutil.rmtree(self._audio_dir / session_id, ignore_errors=True)
        return cursor.rowcount > 0

    def reopen_session(self, session_id: str) -> None:
        """Clear finalization fields so an interrupted session can continue."""
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE sessions
                    SET ended_at = NULL, summary = NULL
                    WHERE id = ?
                    """,
                    (session_id,),
                )

    def add_turn(
        self,
        session_id: str,
        role: str,
        modality: str,
        language: str,
        text: str,
        *,
        audio: bytes | None = None,
        sample_rate: int = 16000,
    ) -> int:
        """Insert a turn and optionally archive raw int16 mono PCM as WAV.

        Args:
            session_id: Existing session id.
            role: `user` or `assistant`.
            modality: `voice` or `text`.
            language: Language tag recorded for the turn.
            text: Transcript text.
            audio: Raw little-endian int16 PCM bytes to archive as a mono WAV.
            sample_rate: WAV framerate used when `audio` is provided.

        Returns:
            The new turn id.

        Raises:
            ValueError: If role, modality, or sample_rate is invalid.
            sqlite3.IntegrityError: If `session_id` does not exist.
        """
        _validate_turn(role, modality, sample_rate)
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO turns (
                        session_id, ts, role, modality, language, text
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, _utc_now(), role, modality, language, text),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return a turn id")
                turn_id = cursor.lastrowid

                if audio is not None:
                    relative_path = Path("audio") / session_id / f"{turn_id}.wav"
                    archive_path = self.data_dir / relative_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_wav(archive_path, audio, sample_rate)
                    connection.execute(
                        "UPDATE turns SET audio_path = ? WHERE id = ?",
                        (relative_path.as_posix(), turn_id),
                    )

        return turn_id

    def sessions(self) -> list[RowDict]:
        """Return all sessions, newest first by start time."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, started_at, ended_at, summary, title
                FROM sessions
                ORDER BY started_at DESC, rowid DESC
                """
            ).fetchall()
        return [_row_dict(row) for row in rows]

    def set_title(self, session_id: str, title: str) -> bool:
        """Set a session's title (user edit — overwrites); False if unknown id."""
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )
        return cursor.rowcount > 0

    def ensure_title(self, session_id: str, title: str) -> None:
        """Fill in an auto-generated title only where none exists yet.

        A user's edit (or an earlier generation) always wins — re-finalizing
        a resumed session must not rename it behind the user's back.
        """
        with self._connect() as connection:
            with connection:
                connection.execute(
                    "UPDATE sessions SET title = COALESCE(title, ?) WHERE id = ?",
                    (title, session_id),
                )

    def session_turns(self, session_id: str) -> list[RowDict]:
        """Return a session's turns in chronological order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, ts, role, modality, language, text, audio_path
                FROM turns
                WHERE session_id = ?
                ORDER BY ts ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [_row_dict(row) for row in rows]

    def recent_summaries(self, limit: int = 10) -> list[RowDict]:
        """Return recent ended-session summaries in reading order."""
        if limit < 0:
            raise ValueError("limit must be >= 0")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT started_at, ended_at, summary
                FROM sessions
                WHERE ended_at IS NOT NULL AND summary IS NOT NULL
                ORDER BY started_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_dict(row) for row in reversed(rows)]

    def upsert_fact(self, statement: str, kind: str = "observation") -> None:
        """Insert or reinforce a canonical-English user-model fact."""
        now = _utc_now()
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO facts (
                        statement, kind, first_seen, last_seen, n_occurrences
                    )
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(statement) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        n_occurrences = facts.n_occurrences + 1
                    """,
                    (statement, kind, now, now),
                )

    def facts(self) -> list[RowDict]:
        """Return all user-model facts in first-seen order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, statement, kind, first_seen, last_seen, n_occurrences
                FROM facts
                ORDER BY first_seen ASC, id ASC
                """
            ).fetchall()
        return [_row_dict(row) for row in rows]

    def export_all(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of sessions, turns, and facts."""
        sessions = []
        for session in self.sessions():
            session_export: dict[str, object] = dict(session)
            session_export["turns"] = self.session_turns(str(session["id"]))
            sessions.append(session_export)
        return {
            "exported_at": _utc_now(),
            "sessions": sessions,
            "facts": self.facts(),
        }

    def delete_all(self) -> None:
        """Delete stored rows and remove the audio archive tree."""
        with self._connect() as connection:
            with connection:
                connection.execute("DELETE FROM facts")
                connection.execute("DELETE FROM turns")
                connection.execute("DELETE FROM sessions")
        shutil.rmtree(self._audio_dir, ignore_errors=True)


def _validate_turn(role: str, modality: str, sample_rate: int) -> None:
    """Validate turn metadata before SQLite or WAV writes."""
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unknown role: {role!r}")
    if modality not in {"voice", "text"}:
        raise ValueError(f"Unknown modality: {modality!r}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")


def _write_wav(path: Path, audio: bytes, sample_rate: int) -> None:
    """Write raw little-endian int16 mono PCM bytes to a WAV file."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
