"""Human gate 1: review inventory records and verify the Document Agent input."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.discover.discover_agent import (
    Inventory, Record, atomic_json, audit, digest, file_sha256, now, output_lock,
    read_sources, validate_evidence,
)
from src.workflow_state import shared_state


def check_sources(record: Record, root: Path) -> None:
    current_sources = []
    for item in record.source_files:
        path = (root / item.path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Source missing or outside input tree: {item.path}")
        if file_sha256(path) != item.sha256:
            raise ValueError(f"Source changed; rerun Discover: {item.path}")
        current_sources.extend(read_sources(path, item.path, item.sha256))
    if [s.model_dump() for s in current_sources] != [s.model_dump() for s in record.sources]:
        raise ValueError("Extracted source text changed; rerun Discover")


def review(output: Path, model_id: str, reviewer: str, decision: str, notes: str) -> Path:
    if not reviewer.strip() or not notes.strip() or decision not in {"approve", "reject"}:
        raise ValueError("Review requires a reviewer, notes and approve/reject decision")
    with output_lock(output):
        inventory = Inventory.model_validate_json((output / "inventory.json").read_text(encoding="utf-8"))
        record = next((r for r in inventory.models if r.model_id == model_id), None)
        if record is None:
            raise ValueError("Unknown model_id")
        if decision == "approve" and record.status == "needs_attention":
            raise ValueError("Resolve scan/extraction errors and rerun before approval")
        check_sources(record, Path(inventory.input_root))
        validate_evidence(record.metadata, record.sources)
        approval = {"reviewer": reviewer, "decision": decision, "notes": notes, "time": now(),
                    "record_sha256": digest(record.model_dump()), "run_id": inventory.run_id}
        # Store each human action immutably; a new discovery run never inherits approval.
        path = output / "reviews" / inventory.run_id / model_id / (str(uuid.uuid4()) + ".json")
        atomic_json(path, {"approval": approval, "inventory_record": record.model_dump(),
                           "input_root": inventory.input_root,
                           "status": "approved_for_document" if decision == "approve" else "rejected"})
        audit(output, {"event": "human_review", "model_id": model_id, **approval, "output_path": str(path)})
    return path


def load_document_input(path: Path) -> dict:
    """Normal entry point for Document Agent; re-check approval and live source hashes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "approved_for_document" or payload["approval"]["decision"] != "approve":
        raise ValueError("Human approval is required")
    record = Record.model_validate(payload["inventory_record"])
    if record.status != "pending_review" or digest(record.model_dump()) != payload["approval"]["record_sha256"]:
        raise ValueError("Record changed after review")
    check_sources(record, Path(payload["input_root"]))
    validate_evidence(record.metadata, record.sources)
    return shared_state({"model_id": record.model_id, "inventory_record": record.model_dump(),
            "input_root": payload["input_root"], "status": "ready_for_document",
            "approvals": [payload["approval"]]})
