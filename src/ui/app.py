"""Gate review console. Run from the repository root:  streamlit run src/ui/app.py

The UI only drives src/pipeline.py: it runs the next stage, shows what the current gate must review, and records
the human decision. Every check and lock lives in the agents, so the UI cannot approve what the CLI would refuse.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # pipeline paths (config/, artifacts/, data/) are relative to the repository root

from src import pipeline  # noqa: E402
from src.ui import dashboard  # noqa: E402

STEP_LABELS = {"discover": "Discover", "gate1": "Gate 1", "document": "Document", "gate2": "Gate 2",
               "redevelop": "Redevelop", "gate3": "Gate 3", "validate": "Validate", "gate4": "Gate 4"}


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def progress_row(state: dict) -> None:
    kind, item = pipeline.next_action(state)
    rebuild = pipeline.config()["models"][state["group"]]["redevelop"]
    steps = [s for s in pipeline.ORDER if rebuild or s in ("discover", "gate1", "document", "gate2")]
    parts = []
    for step in steps:
        info = state["steps"].get(step)
        if info and info.get("decision") == "reject":
            icon = "↩️"
        elif info:
            icon = "✅"
        elif item == step:
            icon = {"run": "▶️", "await": "🧑‍⚖️", "error": "⚠️"}.get(kind, "⬜")
        else:
            icon = "⬜"
        label = f"**{STEP_LABELS[step]}**" if item == step else STEP_LABELS[step]
        parts.append(f"{icon} {label}")
    st.markdown(" → ".join(parts))  # one line that wraps cleanly in narrow windows
    st.caption("✅ done · ▶️ next agent · 🧑‍⚖️ waiting for a human decision · ↩️ rejected, will rerun · ⚠️ failed")


def show_gate1(state: dict) -> None:
    inventory = read_json(state["steps"]["discover"]["inventory_path"])
    record = next(m for m in inventory["models"] if m["group"] == state["group"])
    metadata = record["metadata"]
    st.write(f"**{record['model_id']}** · status `{record['status']}` · extraction `{record['extraction_mode']}` · "
             f"provisional risk tier `{record['risk_tier']}`")
    rows = [[k, ", ".join(map(str, v)) if isinstance(v, list) else str(v if v is not None else ""), ""]
            for k, v in metadata.items() if k not in ("evidence", "unknowns")]
    edited = st.data_editor(pd.DataFrame(rows, columns=["Field", "Extracted value", "Correction"]),
                            hide_index=True, width="stretch", disabled=["Field", "Extracted value"],
                            key=f"corrections-{state['group']}")
    st.session_state["corrections"] = {r["Field"]: r["Correction"] for _, r in edited.iterrows()
                                       if str(r["Correction"] or "").strip()}
    st.caption(f"Fields with source evidence: {', '.join(sorted(metadata['evidence']))}")
    if metadata["unknowns"]:
        st.warning("Unknowns: " + "; ".join(metadata["unknowns"]))
    for warning in record["warnings"]:
        st.error(warning)
    st.info("Type a fix in the Correction column for any wrong field. The hash-locked record is not edited: "
            "corrections are stored with your approval and passed to the Document agent.")


def show_gate2(state: dict) -> None:
    document = read_json(state["steps"]["document"]["state_path"])
    report = read_json(document["consistency_report_path"])
    counts = pd.Series([i["result"] for i in report["items"]]).value_counts().to_dict()
    st.write(f"Status `{document['status']}` · mode `{report['mode']}` · self-check {counts}")
    flagged = [[i["claim_id"], i["result"], i["reason"]] for i in report["items"] if i["result"] != "consistent"]
    if flagged:
        st.subheader("Claims the self-check did not confirm")
        st.dataframe(pd.DataFrame(flagged, columns=["Claim", "Result", "Reason"]), hide_index=True,
                     width="stretch")
    if document.get("failure"):
        st.error(f"Generation failed: {document['failure']}. Reject to regenerate.")
    with st.expander("Draft document", expanded=not flagged):
        st.markdown(Path(document["doc_path"]).read_text(encoding="utf-8"))
    st.info("You may edit the draft file before approving; edits, offline drafts or unconfirmed claims require "
            "resolution notes.")


def show_gate3(state: dict) -> None:
    redevelop = read_json(state["steps"]["redevelop"]["state_path"])
    history = read_json(redevelop["parity_result_path"])["history"]
    st.write(f"Status `{redevelop['status']}` · mode `{redevelop['mode']}` · retries {redevelop['retry_count']}")
    if redevelop["ir_differences_from_baseline"]:
        st.warning("IR differs from the reviewed baseline (resolution notes required):")
        for item in redevelop["ir_differences_from_baseline"]:
            st.code(item)
    rows = []
    for round_ in history:
        checks = "; ".join(f"{c['metric']}={c['value']:.3g}" for c in round_.get("checks", []))
        rows.append([round_["round"], "PASS" if round_["passed"] else "FAIL",
                     checks or round_.get("error", "").splitlines()[0]])
    st.dataframe(pd.DataFrame(rows, columns=["Round", "Result", "Parity checks / error"]), hide_index=True,
                 width="stretch")
    if redevelop.get("difference_report_path"):
        st.error("Retries exhausted.")
        st.markdown(Path(redevelop["difference_report_path"]).read_text(encoding="utf-8"))
    for label, key, language in [("IR", "ir_path", "json"), ("Generated tests", "tests_path", "python"),
                                 ("Generated model code", "generated_code_path", "python")]:
        if redevelop.get(key):
            with st.expander(label):
                st.code(Path(redevelop[key]).read_text(encoding="utf-8"), language=language)


def show_gate4(state: dict) -> None:
    validation = read_json(state["steps"]["validate"]["state_path"])
    report = read_json(validation["metrics_path"])
    st.write(f"Status `{report['status']}` · explainer `{report['explainer_mode']}` · conclusion "
             f"`{report['explanation']['conclusion']}`")
    if report.get("explainer_error"):
        st.warning(f"LLM explainer fell back to the template: {report['explainer_error']}")
    metrics = {k: v for k, v in report["metrics"].items() if k != "calibration"}
    st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]), hide_index=True)
    findings = pd.DataFrame(report["findings"])
    st.dataframe(findings, hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(report["groups"]), hide_index=True, width="stretch")
    for section in ("summary", "performance", "calibration", "stability", "fairness", "conceptual_soundness"):
        st.markdown(f"**{section.replace('_', ' ').title()}.** {report['explanation'][section]}")
    st.markdown("**Limitations:** " + "; ".join(report["explanation"]["limitations"]))
    st.download_button("Download Word report", Path(validation["docx_path"]).read_bytes(),
                       file_name=f"{report['model_id']}_validation.docx")
    st.info("Approval requires resolution notes for the limitations and any unverifiable checks; failed rules "
            "cannot be approved.")


def review_form(group: str, gate: str) -> None:
    with st.form(f"review-{group}-{gate}"):
        reviewer = st.text_input("Reviewer (name or role)", value=st.session_state.get("reviewer", ""))
        decision = st.radio("Decision", ["approve", "reject"], horizontal=True)
        notes = st.text_area("Review notes (required)")
        resolution = st.text_area("Resolution notes (required when edited, unresolved or limitations remain)",
                                  disabled=gate == "gate1")
        if st.form_submit_button(f"Record {pipeline.GATE_NAMES[gate]} decision"):
            st.session_state["reviewer"] = reviewer
            try:
                corrections = st.session_state.get("corrections") if gate == "gate1" else None
                path = pipeline.review(group, reviewer, decision, notes, resolution, corrections)
                st.success(f"Recorded: {path}")
                st.rerun()
            except (ValueError, OSError, KeyError) as error:
                st.error(f"The gate refused this decision: {error}")


def main() -> None:
    pipeline.load_env()
    st.set_page_config(page_title="Model Risk Factory", layout="wide")
    with st.sidebar:
        page = st.radio("View", ["Dashboard", "Gate console"])
    if page == "Dashboard":
        dashboard.render()
        return
    st.title("Agentic Model Risk Factory")
    st.caption("Discover → Document → Redevelop → Validate, with a human gate after each agent. Research use only.")
    groups = list(pipeline.config()["models"])
    with st.sidebar:
        group = st.selectbox("Model", groups)
        has_key = all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"))
        use_llm = st.toggle("Use LLM", value=has_key, disabled=not has_key,
                            help="Needs LLM_BASE_URL, LLM_MODEL and LLM_API_KEY (from .env)")
        st.caption(f"LLM: {os.environ.get('LLM_MODEL') if has_key else 'not configured - offline baselines'}")
    state = pipeline.load_state(group)
    st.subheader(f"{group} · {state['model_id'] or 'not discovered yet'}")
    progress_row(state)
    kind, item = pipeline.next_action(state)
    if state["last_error"] and kind == "run":
        st.error(f"Last run failed at {state['last_error']['stage']}: {state['last_error']['error']}")
    if kind == "run":
        if st.button(f"Run {STEP_LABELS[item]}" + (" with LLM" if use_llm else " offline"), type="primary"):
            with st.spinner(f"Running {item}; LLM stages can take several minutes..."):
                result = pipeline.run(group, use_llm, progress=lambda message: None)
            if result[0] == "error":
                st.error(pipeline.load_state(group)["last_error"]["error"])
            else:
                st.rerun()
    elif kind == "await":
        st.header(pipeline.GATE_NAMES[item])
        {"gate1": show_gate1, "gate2": show_gate2, "gate3": show_gate3, "gate4": show_gate4}[item](state)
        review_form(group, item)
    else:
        last_gate = state["steps"].get("gate4") or state["steps"].get("gate2", {})
        st.success(f"Pipeline complete: {item}. Final decision by {last_gate.get('reviewer')} at "
                   f"{last_gate.get('time')}.")
        if "validate" in state["steps"]:
            st.caption("The figures below are what the gate 4 reviewer approved.")
            show_gate4(state)
    with st.expander("Audit trail"):
        st.dataframe(pd.DataFrame(state["history"]), hide_index=True, width="stretch")


main()
