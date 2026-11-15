"""End-to-end orchestrator: runs agents until the next human gate, then waits for a decision.

A resumable state machine (one JSON file per model) gives the same behaviour as a graph interrupt: `run` executes
stages until a gate, `review` records the human decision, and the next `run` continues. A rejection reruns the
stage before that gate; a rerun clears every later step, so stale approvals can never be reused.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.discover.discover_agent import atomic_json, now, run_discovery
from src.discover.discover_review import review as review_discover
from src.document.document_agent import run_document
from src.document.document_review import load_redevelop_input, review_document
from src.redevelop.redevelop_agent import run_redevelop
from src.redevelop.redevelop_review import review_redevelop
from src.validate.validate_agent import run_validation
from src.validate.validate_review import review_validation

CONFIG = Path("config/pipeline.json")
# PIPELINE_ARTIFACTS lets rehearsals and demos run without touching the shared artifacts/discover/inventory.json.
ARTIFACTS = Path(os.environ.get("PIPELINE_ARTIFACTS", "artifacts"))
STAGES = [("discover", "gate1"), ("document", "gate2"), ("redevelop", "gate3"), ("validate", "gate4")]
ORDER = [step for pair in STAGES for step in pair]
GATE_NAMES = {"gate1": "Discover review", "gate2": "Document review", "gate3": "Redevelop review",
              "gate4": "Validate review"}


def load_env(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines into this process only; existing variables win; nothing is printed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() and value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def state_path(group: str) -> Path:
    return ARTIFACTS / "pipeline" / f"{group}.json"


def load_state(group: str) -> dict:
    if group not in config()["models"]:
        raise ValueError(f"Unknown model group {group}; configure it in {CONFIG}")
    path = state_path(group)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"group": group, "model_id": None, "steps": {}, "history": [], "last_error": None}


def save_state(state: dict) -> None:
    atomic_json(state_path(state["group"]), state)


def next_action(state: dict) -> tuple[str, str]:
    """('run', stage), ('await', gate) or ('done', outcome)."""
    redevelop = config()["models"][state["group"]]["redevelop"]
    steps = state["steps"]
    for stage, gate in STAGES:
        if stage == "redevelop" and not redevelop:
            return "done", "documented"
        if stage not in steps:
            return "run", stage
        if gate not in steps:
            return "await", gate
        if steps[gate]["decision"] == "reject":
            return "run", stage
    return "done", "validated_for_research"


def _clear_from(state: dict, step: str) -> None:
    for later in ORDER[ORDER.index(step):]:
        state["steps"].pop(later, None)


def _event(state: dict, **event) -> None:
    state["history"].append({"time": now(), **event})


def run_stage(state: dict, stage: str, use_llm: bool, caller=None) -> None:
    steps, group = state["steps"], state["group"]
    rejected = steps.get(dict(STAGES)[stage], {})
    feedback = rejected.get("review_path") if rejected.get("decision") == "reject" else None
    _clear_from(state, stage)
    if stage == "discover":
        inventory = run_discovery(Path(config()["legacy_root"]), ARTIFACTS / "discover", use_llm)
        record = next((m for m in inventory.models if m.group == group), None)
        if record is None:
            raise ValueError(f"Discover found no model group {group}")
        state["model_id"] = record.model_id
        steps["discover"] = {"inventory_run_id": inventory.run_id, "record_status": record.status,
                             "inventory_path": str(ARTIFACTS / "discover" / "runs" / inventory.run_id / "inventory.json")}
    elif stage == "document":
        path = run_document(Path(steps["gate1"]["review_path"]), ARTIFACTS / "document", use_llm=use_llm,
                            caller=caller, feedback_review=Path(feedback) if feedback and use_llm else None,
                            max_calls=60)
        steps["document"] = {"state_path": str(path), "status": json.loads(path.read_text(encoding="utf-8"))["status"]}
    elif stage == "redevelop":
        settings = config()["models"][group]["redevelop"]
        gate2 = Path(steps["gate2"]["review_path"])
        upstream = {**load_redevelop_input(gate2), "document_review_path": str(gate2.resolve())}
        final = run_redevelop(upstream, Path(settings["data"]), Path(settings["reference"]), ARTIFACTS / "redevelop",
                              settings["tier"], caller=caller, offline=not use_llm, baseline=settings.get("baseline"))
        steps["redevelop"] = {"state_path": str(Path(final["run_dir"]) / "state.json"), "status": final["status"],
                              "retry_count": final["retry_count"]}
    elif stage == "validate":
        policy = config()["models"][group].get("validation_policy")
        path = run_validation(Path(steps["gate3"]["review_path"]), ARTIFACTS / "validate",
                              **({"policy_path": Path(policy)} if policy else {}), use_llm=use_llm, caller=caller)
        steps["validate"] = {"state_path": str(path), "status": json.loads(path.read_text(encoding="utf-8"))["status"]}
    _event(state, event="stage_completed", stage=stage, llm=use_llm, status=steps[stage].get("status"))


def run(group: str, use_llm: bool, caller=None, progress=print) -> tuple[str, str]:
    """Run stages until a gate needs a human (or the pipeline is done or a stage fails)."""
    state = load_state(group)
    while True:
        kind, item = next_action(state)
        if kind != "run":
            save_state(state)
            return kind, item
        progress(f"[{group}] running {item}{' with LLM' if use_llm else ' offline'} ...")
        try:
            run_stage(state, item, use_llm, caller)
            state["last_error"] = None
        except Exception as error:  # surface any agent failure to the operator; nothing is auto-approved
            state["last_error"] = {"stage": item, "error": f"{type(error).__name__}: {error}"[:1000], "time": now()}
            _event(state, event="stage_failed", stage=item, error=state["last_error"]["error"])
            save_state(state)
            return "error", item
        save_state(state)


def registry_path() -> Path:
    return ARTIFACTS / "registry" / "models.json"


def load_registry() -> dict:
    path = registry_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"models": {}}


def register(state: dict, stage: str, gate: str, review_path: Path) -> None:
    """Local model registry (stand-in for MLflow Model Registry): gate approvals move a rebuild's stage."""
    registry = load_registry()
    code = json.loads(Path(state["steps"]["redevelop"]["state_path"]).read_text(encoding="utf-8"))
    entry = registry["models"].setdefault(state["model_id"], {"group": state["group"], "history": []})
    entry.update(stage=stage, version=code["code_sha256"], code_path=code["generated_code_path"], updated_at=now())
    entry["history"].append({"stage": stage, "gate": gate, "review_path": str(review_path), "time": now(),
                             "version": code["code_sha256"]})
    atomic_json(registry_path(), registry)


def review(group: str, reviewer: str, decision: str, notes: str, resolution_notes: str = "",
           corrections: dict[str, str] | None = None, return_to_review1: bool = False, issue: str = "") -> Path:
    """Record a human decision at the gate the pipeline is waiting for.

    return_to_review1: a gate 3/4 rejection whose root cause is the approved document reopens gate 2 (Review 1)
    with the named issue, instead of rerunning the same stage on the same document.
    """
    state = load_state(group)
    kind, gate = next_action(state)
    if kind != "await":
        raise ValueError(f"No gate is waiting for {group} (next action: {kind} {gate})")
    if return_to_review1 and (decision != "reject" or gate not in ("gate3", "gate4")):
        raise ValueError("Return to Review 1 is only a gate 3 or gate 4 rejection")
    if return_to_review1 and not issue.strip():
        raise ValueError("Return to Review 1 needs the issue: which point of the approved document looks wrong")
    steps = state["steps"]
    if gate == "gate1":
        path = review_discover(ARTIFACTS / "discover", state["model_id"], reviewer, decision, notes, corrections)
    elif gate == "gate2":
        path = review_document(Path(steps["document"]["state_path"]), reviewer, decision, notes, resolution_notes)
    elif gate == "gate3":
        path = review_redevelop(Path(steps["redevelop"]["state_path"]), reviewer, decision, notes, resolution_notes)
    else:
        path = review_validation(Path(steps["validate"]["state_path"]), reviewer, decision, notes, resolution_notes)
    steps[gate] = {"review_path": str(path), "decision": decision, "reviewer": reviewer, "time": now()}
    _event(state, event="gate_decision", gate=gate, decision=decision, reviewer=reviewer)
    if gate == "gate3" and decision == "approve":
        register(state, "Staging", gate, path)
    elif gate == "gate4":
        register(state, "Validated (research)" if decision == "approve" else "Rejected", gate, path)
    if return_to_review1:
        _clear_from(state, "gate2")  # the rebuild came from the document under question, so it is redone too
        state["review1_issue"] = {"from_gate": gate, "issue": issue, "reviewer": reviewer,
                                  "review_path": str(path), "time": now()}
        _event(state, event="returned_to_review1", gate=gate, issue=issue, review_path=str(path))
    elif gate == "gate2" and decision == "approve":
        state.pop("review1_issue", None)
    save_state(state)
    return path


def describe(group: str) -> str:
    state = load_state(group)
    kind, item = next_action(state)
    lines = [f"{group} ({state['model_id'] or 'not discovered'}): next = {kind} {GATE_NAMES.get(item, item)}"]
    for step in ORDER:
        if step in state["steps"]:
            info = state["steps"][step]
            lines.append(f"  {step:9} {info.get('decision') or info.get('status') or info.get('record_status', '')}")
    if state["last_error"]:
        lines.append(f"  last error at {state['last_error']['stage']}: {state['last_error']['error'][:200]}")
    return "\n".join(lines)


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--model", help="Model group; all configured groups if omitted")
    go = sub.add_parser("run")
    go.add_argument("--model", required=True)
    go.add_argument("--llm", action="store_true", help="Use the configured LLM (loaded from .env if present)")
    gate = sub.add_parser("review")
    gate.add_argument("--model", required=True)
    gate.add_argument("--reviewer", required=True)
    gate.add_argument("--decision", choices=["approve", "reject"], required=True)
    gate.add_argument("--notes", required=True)
    gate.add_argument("--resolution-notes", default="")
    gate.add_argument("--correct", action="append", default=[], metavar="FIELD=VALUE",
                      help="Gate 1 only: reviewer correction to an extracted field; repeatable")
    gate.add_argument("--return-to-review1", action="store_true",
                      help="Gate 3/4 rejection only: the approved document is wrong; reopen gate 2")
    gate.add_argument("--issue", default="", help="With --return-to-review1: the point Review 1 must check")
    sub.add_parser("registry")
    args = parser.parse_args()
    try:
        if args.command == "status":
            for group in [args.model] if args.model else config()["models"]:
                print(describe(group))
        elif args.command == "run":
            kind, item = run(args.model, args.llm, progress=lambda m: print(m, file=sys.stderr, flush=True))
            print(describe(args.model))
            return 1 if kind == "error" else 0
        elif args.command == "registry":
            for model_id, entry in load_registry()["models"].items():
                print(f"{model_id} ({entry['group']}): {entry['stage']}  version {entry['version'][:12]}  "
                      f"updated {entry['updated_at']}")
        else:
            corrections = dict(item.split("=", 1) for item in args.correct if "=" in item)
            print(review(args.model, args.reviewer, args.decision, args.notes, args.resolution_notes, corrections,
                         args.return_to_review1, args.issue))
            print(describe(args.model))
    except (ValueError, OSError, KeyError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
