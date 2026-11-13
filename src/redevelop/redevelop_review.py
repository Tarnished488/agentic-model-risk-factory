"""Human gate 3: review the IR, generated code, tests and parity, snapshot them, and hand off to Validate."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from src.discover.discover_agent import atomic_json, audit, digest, file_sha256, now, output_lock
from src.document.document_review import load_redevelop_input

ARTIFACTS = {"ir": "ir_path", "code": "generated_code_path", "tests": "tests_path"}


HANDOFF = ("data_path", "train_ids_path", "holdout_predictions_path", "coefficients_path")


def _texts(state: dict) -> dict[str, str]:
    return {name: Path(state[key]).read_text(encoding="utf-8") for name, key in ARTIFACTS.items()}


def _check_handoff_files(state: dict) -> None:
    for key, expected in state["handoff_sha256"].items():
        if file_sha256(Path(state[key])) != expected:
            raise ValueError(f"{key} changed after the Redevelop run")


def load_redevelop_state(state_path: Path) -> dict:
    """Re-verify that what the reviewer sees is exactly what the agent produced."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] not in {"pending_redevelop_review", "needs_attention"}:
        raise ValueError("Not a reviewable Redevelop state")
    parity = json.loads(Path(state["parity_result_path"]).read_text(encoding="utf-8"))
    if digest(parity) != state["parity_sha256"] or parity["status"] != state["status"]:
        raise ValueError("Parity result changed after the Redevelop run")
    if state["generated_code_path"] is None:
        raise ValueError("No generated code to review")
    texts = _texts(state)
    if (digest(json.loads(texts["ir"])) != state["ir_sha256"] or digest(texts["code"]) != state["code_sha256"]
            or digest(texts["tests"]) != state["tests_sha256"]):
        raise ValueError("IR, code or tests changed after the Redevelop run")
    _check_handoff_files(state)
    return state


def review_redevelop(state_path: Path, reviewer: str, decision: str, notes: str,
                     resolution_notes: str = "") -> Path:
    if not reviewer.strip() or not notes.strip() or decision not in {"approve", "reject"}:
        raise ValueError("Review requires reviewer, notes and approve/reject decision")
    with output_lock(state_path.resolve().parent):
        state = load_redevelop_state(state_path)
        if decision == "approve":
            if state["status"] != "pending_redevelop_review" or not state["parity_result"]["passed"]:
                raise ValueError("Only a run that passed parity can be approved; escalated runs need a rerun")
            if state["ir_differences_from_baseline"] and not resolution_notes.strip():
                raise ValueError("IR differs from the reviewed baseline: explain the resolution before approval")
        review_id = str(uuid.uuid4())
        directory = state_path.resolve().parent / "reviews" / review_id
        directory.mkdir(parents=True)
        texts = _texts(state)
        locked = {"ir": directory / "ir.json", "code": directory / "new_model.py",
                  "tests": directory / "test_new_model.py"}
        for name, path in locked.items():
            path.write_text(texts[name], encoding="utf-8")
        approval = {"gate": "redevelop", "review_id": review_id, "reviewer": reviewer, "decision": decision,
                    "notes": notes, "resolution_notes": resolution_notes, "time": now(),
                    "run_id": state["redevelop_run_id"], "ir_sha256": state["ir_sha256"],
                    "code_sha256": state["code_sha256"], "tests_sha256": state["tests_sha256"],
                    "parity_sha256": state["parity_sha256"],
                    "ir_differences": state["ir_differences_from_baseline"]}
        snapshot = {**state, "ir_path": str(locked["ir"]), "generated_code_path": str(locked["code"]),
                    "tests_path": str(locked["tests"]),
                    "status": "approved_for_validate" if decision == "approve" else "rejected_redevelop",
                    # Stand-in for a model registry: approval promotes the rebuild to Staging.
                    "registry_stage": "Staging" if decision == "approve" else None,
                    "approvals": [*state["approvals"], approval], "approval": approval,
                    "approved_texts": texts, "rejection_feedback": notes if decision == "reject" else None}
        snapshot["snapshot_sha256"] = digest(snapshot)
        path = directory / "review.json"
        atomic_json(path, snapshot)
        audit(state_path.resolve().parent, {"node": "redevelop", "event": "human_review",
                                            "model_id": state["model_id"], "output_path": str(path), **approval})
        return path


def load_validate_input(path: Path) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    checksum = snapshot.pop("snapshot_sha256", None)
    if checksum != digest(snapshot):
        raise ValueError("Redevelop review snapshot changed after approval")
    approval = snapshot["approval"]
    if snapshot["status"] != "approved_for_validate" or approval["decision"] != "approve":
        raise ValueError("Redevelop human approval is required")
    texts = _texts(snapshot)
    if texts != snapshot["approved_texts"] or digest(texts["code"]) != approval["code_sha256"]:
        raise ValueError("Approved IR, code or tests changed")
    _check_handoff_files(snapshot)
    if not snapshot.get("document_review_path"):
        raise ValueError("Redevelop run has no gate-2 lineage; rerun it from an approved document review")
    upstream = load_redevelop_input(Path(snapshot["document_review_path"]))
    if snapshot["approvals"][:-1] != upstream["approvals"] or snapshot["model_id"] != upstream["model_id"]:
        raise ValueError("Approved Redevelop lineage differs from the approved document")
    return {**upstream, "ir_path": snapshot["ir_path"], "generated_code_path": snapshot["generated_code_path"],
            "tests_path": snapshot["tests_path"], "parity_result": snapshot["parity_result"],
            **{key: snapshot.get(key) for key in HANDOFF}, "handoff_sha256": snapshot["handoff_sha256"],
            "retry_count": snapshot["retry_count"], "registry_stage": snapshot["registry_stage"],
            "approvals": snapshot["approvals"], "status": "ready_for_validate"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("review")
    gate.add_argument("state", type=Path, help="artifacts/redevelop/runs/<run_id>/state.json")
    gate.add_argument("--reviewer", required=True)
    gate.add_argument("--decision", choices=["approve", "reject"], required=True)
    gate.add_argument("--notes", required=True)
    gate.add_argument("--resolution-notes", default="")
    handoff = sub.add_parser("handoff")
    handoff.add_argument("review_file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "review":
            print(review_redevelop(args.state, args.reviewer, args.decision, args.notes, args.resolution_notes))
        else:
            print(json.dumps(load_validate_input(args.review_file), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        print(f"Redevelop review error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
