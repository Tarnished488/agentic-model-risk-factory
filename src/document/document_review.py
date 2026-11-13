"""Human gate 2: review, snapshot the edited document, and verify Redevelop input."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from src.discover.discover_agent import atomic_json, audit, digest, now, output_lock
from src.discover.discover_review import load_document_input


def load_draft(state_path: Path):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] not in {"pending_document_review", "needs_attention"}:
        raise ValueError("Not a reviewable Document state")
    upstream = load_document_input(Path(state["discover_review_path"]))
    for key in ("model_id", "inventory_record", "input_root", "approvals"):
        if state[key] != upstream[key]:
            raise ValueError("Document lineage differs from approved Discover input")
    report = json.loads(Path(state["consistency_report_path"]).read_text(encoding="utf-8"))
    if digest(report) != state["consistency_report_sha256"]:
        raise ValueError("Consistency report changed after generation")
    if (report["record_sha256"] != digest(state["inventory_record"])
            or report["run_id"] != state["document_run_id"]
            or report["model_id"] != state["model_id"]
            or report["document_sha256"] != state["document_sha256"]
            or report["status"] != state["status"] or report["failure"] != state["failure"]):
        raise ValueError("Document report and state disagree")
    return state, report


def review_document(state_path: Path, reviewer: str, decision: str, notes: str,
                    resolution_notes: str = "") -> Path:
    if not reviewer.strip() or not notes.strip() or decision not in {"approve", "reject"}:
        raise ValueError("Review requires reviewer, notes and approve/reject decision")
    # state_path is .../runs/<uuid>/state.json; use the run's own review directory.
    with output_lock(state_path.resolve().parent):
        state, report = load_draft(state_path)
        text = Path(state["doc_path"]).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Cannot review an empty document")
        edited = digest(text) != state["document_sha256"]
        unresolved = report["mode"] != "llm" or any(i["result"] != "consistent" for i in report["items"])
        if decision == "approve":
            if report["failure"]:
                raise ValueError("Resolve failed generation/self-check and rerun before approval")
            if (edited or unresolved) and not resolution_notes.strip():
                raise ValueError("Edited/offline/unresolved document requires explicit human resolution notes")
        review_id = str(uuid.uuid4())
        directory = state_path.resolve().parent / "reviews" / review_id
        directory.mkdir(parents=True)
        locked_doc = directory / "approved_document.md"
        locked_doc.write_text(text, encoding="utf-8")
        approval = {"gate": "document", "review_id": review_id, "reviewer": reviewer,
                    "decision": decision, "notes": notes, "resolution_notes": resolution_notes,
                    "time": now(), "run_id": state["document_run_id"],
                    "record_sha256": digest(state["inventory_record"]),
                    "original_document_sha256": state["document_sha256"],
                    "document_sha256": digest(text), "human_edited": edited,
                    "consistency_report_sha256": digest(report)}
        snapshot = {**state, "doc_path": str(locked_doc), "document_sha256": digest(text),
                    "status": "approved_for_redevelop" if decision == "approve" else "rejected_document",
                    "approvals": [*state["approvals"], approval], "approval": approval,
                    "approved_document_text": text, "consistency_report": report,
                    "rejection_feedback": notes if decision == "reject" else None}
        snapshot["snapshot_sha256"] = digest(snapshot)
        path = directory / "review.json"
        atomic_json(path, snapshot)
        audit(state_path.resolve().parent, {"node": "document", "event": "human_review",
              "model_id": state["model_id"], "output_path": str(path), **approval})
        return path


def load_redevelop_input(path: Path) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    checksum = snapshot.pop("snapshot_sha256", None)
    if checksum != digest(snapshot):
        raise ValueError("Document review snapshot changed after approval")
    approval = snapshot["approval"]
    if snapshot["status"] != "approved_for_redevelop" or approval["decision"] != "approve":
        raise ValueError("Document human approval is required")
    upstream = load_document_input(Path(snapshot["discover_review_path"]))
    if (snapshot["inventory_record"] != upstream["inventory_record"]
            or snapshot["model_id"] != upstream["model_id"]
            or snapshot["approvals"][:-1] != upstream["approvals"]
            or snapshot["approvals"][-1] != approval):
        raise ValueError("Approved document lineage changed")
    text = Path(snapshot["doc_path"]).read_text(encoding="utf-8")
    if (digest(text) != approval["document_sha256"] or text != snapshot["approved_document_text"]
            or digest(snapshot["consistency_report"]) != approval["consistency_report_sha256"]
            or digest(snapshot["inventory_record"]) != approval["record_sha256"]):
        raise ValueError("Approved document, report or model record changed")
    return {**upstream, "doc_path": snapshot["doc_path"], "document_text": text,
            "document_sha256": approval["document_sha256"],
            "consistency_report": snapshot["consistency_report"],
            "approvals": snapshot["approvals"], "status": "ready_for_redevelop",
            "retry_count": snapshot["retry_count"], "document_run_id": snapshot["document_run_id"]}


def load_rejection_feedback(path: Path, state: dict) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    checksum = snapshot.pop("snapshot_sha256", None)
    if checksum != digest(snapshot):
        raise ValueError("Rejected review snapshot changed")
    if snapshot["status"] != "rejected_document" or snapshot["approval"]["decision"] != "reject":
        raise ValueError("Feedback must come from a rejected Document review")
    if (snapshot["model_id"] != state["model_id"]
            or snapshot["inventory_record"] != state["inventory_record"]
            or snapshot["approvals"][:-1] != state["approvals"]):
        raise ValueError("Rejection feedback belongs to another model or Discover version")
    return {"review_path": str(path.resolve()), "approval": snapshot["approval"],
            "feedback": snapshot["rejection_feedback"]}
