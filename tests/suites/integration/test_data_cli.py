"""Complete CLI/API owner-data export, restore, and erasure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from therapy.data import DataSovereignty
from therapy.dialogue.outreach import ProactivityService
from therapy.knowledge.embeddings import EmbeddingMetadata
from therapy.knowledge.research import ResearchKB
from therapy.knowledge.user_model import UserModel
from therapy.memory import MemoryStore
from therapy.memory import __main__ as cli_module
from therapy.memory.__main__ import main


class ExportEmbedder:
    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata("export-test", "v1", 2)

    @staticmethod
    def _vector(_text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def _seed(data_dir: Path) -> tuple[str, int]:
    store = MemoryStore(data_dir)
    session_id = store.create_session()
    store.add_turn(
        session_id,
        "user",
        "voice",
        "en",
        "I journaled today.",
        audio=b"\x00\x00\x01\x00",
    )
    store.end_session(session_id, "The user journaled today.")
    store.ensure_recap(session_id, "You journaled.")
    model = UserModel(data_dir)
    node_id = model.add_user_statement("identity_fact", "The user journals.")
    assert node_id is not None
    model.add_boundary("never_initiate", "private topic")
    research = ResearchKB(data_dir, embedder=ExportEmbedder())
    research.ingest("Journaling paper", "Doe 2025", "Journaling supports planning.")
    outreach = ProactivityService(data_dir, model=model, push_sender=lambda *_: None)
    outreach.update_settings(
        "greeting",
        enabled=True,
        timezone="UTC",
        quiet_start="22:00",
        quiet_end="08:00",
        schedule_time="18:00",
        schedule_day=6,
        frequency="weekly",
        topic=None,
    )
    return session_id, node_id


def test_cli_export_contains_all_tables_audio_corpus_and_private_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    session_id, _ = _seed(tmp_path)
    output = tmp_path / "owner-export.json"

    assert main(["export", "--output", str(output)]) == 0

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["format"] == "therapy-owner-data"
    assert snapshot["tables"]["sessions"][0]["id"] == session_id
    assert snapshot["tables"]["nodes"][0]["statement"] == "The user journals."
    assert snapshot["tables"]["research_documents"][0]["source_title"] == "Journaling paper"
    assert any(
        row["channel"] == "greeting" and row["enabled"] == 1
        for row in snapshot["tables"]["proactivity_settings"]
    )
    assert any(file["path"].endswith(".wav") for file in snapshot["files"])
    assert any(file["path"].startswith("research/sources/") for file in snapshot["files"])
    assert os.stat(output).st_mode & 0o777 == 0o600


def test_cli_delete_requires_confirmation_and_then_erases_every_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    _seed(tmp_path)
    (tmp_path / "therapy.db.pre-phase4-test.bak").write_bytes(b"private backup")

    assert main(["delete"]) == 2
    assert MemoryStore(tmp_path).sessions()
    assert main(["delete", "--yes"]) == 0

    assert "owner data and corpus" in capsys.readouterr().err
    assert MemoryStore(tmp_path).sessions() == []
    assert UserModel(tmp_path).nodes() == []
    assert ResearchKB(tmp_path, embedder=ExportEmbedder()).documents() == []
    assert all(not item["enabled"] for item in ProactivityService(tmp_path).settings())
    assert not (tmp_path / "audio").exists()
    assert not list((tmp_path / "research" / "sources").glob("*"))
    assert not list(tmp_path.glob("therapy.db*.bak"))


def test_cli_restore_round_trip_recovers_rows_audio_keys_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    session_id, node_id = _seed(tmp_path)
    vapid_public_key = ProactivityService(tmp_path).vapid.public_key()
    output = tmp_path / "snapshot.json"
    assert main(["export", "--output", str(output)]) == 0
    assert main(["delete", "--yes"]) == 0

    assert main(["restore", "--input", str(output)]) == 2
    assert main(["restore", "--input", str(output), "--yes"]) == 0

    store = MemoryStore(tmp_path)
    assert store.sessions()[0]["id"] == session_id
    assert store.turn_audio_path(session_id, store.session_turns(session_id)[0]["id"])
    assert UserModel(tmp_path).get_node(node_id)["statement"] == "The user journals."
    assert ResearchKB(tmp_path, embedder=ExportEmbedder()).documents()[0]["source_ref"] == "Doe 2025"
    assert ProactivityService(tmp_path).vapid.public_key() == vapid_public_key
    greeting = next(
        item for item in ProactivityService(tmp_path).settings() if item["channel"] == "greeting"
    )
    assert greeting["enabled"] is True


def test_invalid_restore_is_rejected_before_current_data_changes(tmp_path: Path) -> None:
    session_id, _ = _seed(tmp_path)
    service = DataSovereignty(tmp_path)
    snapshot = service.export_snapshot()
    snapshot["tables"]["nodes"][0]["unknown_column"] = "bad"

    with pytest.raises(ValueError, match="invalid columns"):
        service.restore_snapshot(snapshot)

    assert MemoryStore(tmp_path).sessions()[0]["id"] == session_id


def test_restore_rejects_incompatible_knowledge_schema(tmp_path: Path) -> None:
    session_id, _ = _seed(tmp_path)
    service = DataSovereignty(tmp_path)
    snapshot = service.export_snapshot()
    snapshot["knowledge_schema_version"] = 999

    with pytest.raises(ValueError, match="knowledge schema version"):
        service.restore_snapshot(snapshot)

    assert MemoryStore(tmp_path).sessions()[0]["id"] == session_id


def test_api_export_delete_and_restore_round_trip(data_dir: Path, client) -> None:
    session_id, _ = _seed(data_dir)
    exported = client.get("/api/data/export")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]

    assert client.request("DELETE", "/api/data", json={"confirmation": "no"}).status_code == 422
    assert client.request(
        "DELETE", "/api/data", json={"confirmation": "DELETE EVERYTHING"}
    ).status_code == 200
    assert MemoryStore(data_dir).sessions() == []

    restored = client.post(
        "/api/data/restore",
        files={"file": ("snapshot.json", exported.content, "application/json")},
    )
    assert restored.status_code == 200
    assert MemoryStore(data_dir).sessions()[0]["id"] == session_id


def test_research_cli_ingest_preview_correct_reindex_list_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        cli_module,
        "ResearchKB",
        lambda: ResearchKB(tmp_path, embedder=ExportEmbedder()),
    )
    source = tmp_path / "planning.md"
    source.write_text(
        "# Planning transitions\n\nA visible checklist reduces planning load.",
        encoding="utf-8",
    )

    assert main(
        [
            "research-ingest",
            str(source),
            "--title",
            "Planning transitions",
            "--ref",
            "Owner guide",
        ]
    ) == 0
    document_id = json.loads(capsys.readouterr().out)["document_id"]

    assert main(["research-list"]) == 0
    listed = json.loads(capsys.readouterr().out)["documents"]
    assert listed[0]["id"] == document_id

    assert main(["research-show", str(document_id)]) == 0
    document = json.loads(capsys.readouterr().out)["document"]
    anchor = document["blocks"][0]["anchor"]
    assert anchor == "section-planning-transitions-block-1"

    corrected = "A visible checklist can reduce planning load."
    assert main(
        ["research-correct", str(document_id), anchor, "--text", corrected]
    ) == 0
    assert json.loads(capsys.readouterr().out)["document"]["blocks"][0]["text"] == corrected

    assert main(["research-reindex", str(document_id)]) == 0
    assert json.loads(capsys.readouterr().out)["chunks_indexed"] == 1
    assert main(["research-delete", str(document_id)]) == 2
    assert main(["research-delete", str(document_id), "--yes"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == document_id
    assert ResearchKB(tmp_path, embedder=ExportEmbedder()).documents() == []
