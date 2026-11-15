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
from src.ui import dashboard, style  # noqa: E402

STEP_LABELS = {"discover": "Discover", "gate1": "Gate 1", "document": "Document", "gate2": "Gate 2",
               "redevelop": "Redevelop", "gate3": "Gate 3", "validate": "Validate", "gate4": "Gate 4"}


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def progress_row(state: dict) -> None:
    """Numbered stepper: the eight steps are a real sequence, so the numbers carry meaning."""
    kind, item = pipeline.next_action(state)
    rebuild = pipeline.config()["models"][state["group"]]["redevelop"]
    steps = [s for s in pipeline.ORDER if rebuild or s in ("discover", "gate1", "document", "gate2")]
    cells = []
    for number, step in enumerate(steps, 1):
        info = state["steps"].get(step)
        if info and info.get("decision") == "reject":
            css, status = "rejected", "Rejected, rerunning"
        elif info:
            css, status = "done", "Approved" if step.startswith("gate") else "Done"
        elif item == step:
            css, status = {"run": ("current", "Next"), "await": ("current waiting", "Your decision"),
                           "error": ("current", "Failed")}.get(kind, ("", ""))
        else:
            css, status = "", ""
        cells.append(f"<div class='{css}'><span class='n'>{number:02d}</span>{STEP_LABELS[step]}"
                     f"{'<br><small>' + status + '</small>' if status else ''}</div>")
    st.markdown(f"<nav class='stepper' aria-label='Pipeline progress'>{''.join(cells)}</nav>",
                unsafe_allow_html=True)


def show_gate1(state: dict) -> None:
    inventory = read_json(state["steps"]["discover"]["inventory_path"])
    record = next(m for m in inventory["models"] if m["group"] == state["group"])
    metadata = record["metadata"]
    style.note("Check each extracted field against the source files. Type a fix in the Correction column for "
               "anything wrong: the hash-locked record is not edited, and your corrections travel with the approval "
               "to the Document agent.")
    st.markdown(f"Extraction `{record['extraction_mode']}` · status `{record['status']}` · provisional risk tier "
                f"`{record['risk_tier']}`")
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


def show_gate2(state: dict) -> None:
    document = read_json(state["steps"]["document"]["state_path"])
    report = read_json(document["consistency_report_path"])
    counts = pd.Series([i["result"] for i in report["items"]]).value_counts().to_dict()
    style.note("Read the draft against the source. The self-check can be lenient, so do not rely on it alone. "
               "You may edit the draft file before approving; edits, offline drafts or unconfirmed claims need "
               "resolution notes.")
    style.kpi_strip([(name, str(counts.get(name, 0))) for name in ("consistent", "inconsistent", "unverifiable")]
                    + [("generation mode", report["mode"])])
    flagged = [[i["claim_id"], i["result"], i["reason"]] for i in report["items"] if i["result"] != "consistent"]
    if flagged:
        st.subheader("Claims the self-check did not confirm")
        st.dataframe(pd.DataFrame(flagged, columns=["Claim", "Result", "Reason"]), hide_index=True,
                     width="stretch")
    if document.get("failure"):
        st.error(f"Generation failed: {document['failure']}. Reject to regenerate.")
    with st.expander("Draft document", expanded=not flagged):
        st.markdown(Path(document["doc_path"]).read_text(encoding="utf-8"))


def show_gate3(state: dict) -> None:
    redevelop = read_json(state["steps"]["redevelop"]["state_path"])
    history = read_json(redevelop["parity_result_path"])["history"]
    style.note("Compare the IR and generated code with the legacy source, and confirm parity passed. Approval "
               "registers the rebuild as Staging.")
    style.kpi_strip([("rebuild mode", redevelop["mode"]), ("retries", str(redevelop["retry_count"])),
                     ("parity rounds", str(len(history))),
                     ("IR differences", str(len(redevelop["ir_differences_from_baseline"] or [])))])
    st.caption("Migration scope: " + redevelop.get("migration_scope", "reestimation"))
    if redevelop["ir_differences_from_baseline"]:
        st.warning("IR differs from the reviewed baseline (resolution notes required):")
        for item in redevelop["ir_differences_from_baseline"]:
            st.code(item)
    rows = []
    for round_ in history:
        checks = "; ".join(f"{c['metric']}={c['value']:.3g}" for c in round_.get("checks", []))
        rows.append([round_["round"], "Pass" if round_["passed"] else "Fail",
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
    counts = pd.Series([f.get("rating", "") for f in report["findings"]]).value_counts().to_dict()
    style.rating_band(report.get("overall_rating", "n/a"), counts)
    if report.get("explainer_error"):
        st.warning(f"The LLM explanation failed its number check, so the template was used: "
                   f"{report['explainer_error']}")
    st.markdown(f"**Explanation** · {report['explainer_mode']} · conclusion "
                f"`{report['explanation']['conclusion']}`")
    for section in ("summary", "performance", "calibration", "stability", "fairness", "conceptual_soundness"):
        st.markdown(f"**{section.replace('_', ' ').capitalize()}.** {report['explanation'][section]}")
    dashboard.charts(report)
    st.subheader("Findings, worst first")
    dashboard.findings_table(report)
    st.markdown("**Limitations.** " + " ".join(report["explanation"]["limitations"]))
    st.download_button("Download the Word validation report", Path(validation["docx_path"]).read_bytes(),
                       file_name=f"{report['model_id']}_validation.docx")
    if state["steps"].get("gate4") is None:
        style.note("Red checks cannot be approved. Approval needs resolution notes for amber items and the "
                   "listed limitations; it marks the model validated for research only.")


def review_form(group: str, gate: str) -> None:
    st.subheader("Your decision")
    with st.form(f"review-{group}-{gate}"):
        reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer", ""),
                                 placeholder="Your name or role, e.g. Validate reviewer")
        decision = st.radio("Decision", ["approve", "reject"], horizontal=True,
                            format_func=lambda d: {"approve": "Approve", "reject": "Reject and rerun"}[d])
        notes = st.text_area("Review notes", placeholder="What you checked and what you found")
        resolution = st.text_area("Resolution notes", disabled=gate == "gate1",
                                  placeholder="How each open item, edit or limitation is resolved")
        if st.form_submit_button(f"Record decision for {STEP_LABELS[gate]}", type="primary"):
            st.session_state["reviewer"] = reviewer
            try:
                corrections = st.session_state.get("corrections") if gate == "gate1" else None
                path = pipeline.review(group, reviewer, decision, notes, resolution, corrections)
                st.success(f"Recorded: {path}")
                st.rerun()
            except (ValueError, OSError, KeyError) as error:
                st.error(f"Not recorded: {error}. Address this and submit again.")


def main() -> None:
    pipeline.load_env()
    st.set_page_config(page_title="Model risk control ledger", layout="wide")
    style.inject()
    with st.sidebar:
        st.markdown("### Model risk factory")
        page = st.radio("View", ["Dashboard", "Gate console"], label_visibility="collapsed")
    if page == "Dashboard":
        dashboard.render()
        return
    groups = list(pipeline.config()["models"])
    with st.sidebar:
        st.divider()
        group = st.selectbox("Model", groups)
        has_key = all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"))
        use_llm = st.toggle("Use the LLM", value=has_key, disabled=not has_key,
                            help="Needs LLM_BASE_URL, LLM_MODEL and LLM_API_KEY in .env")
        st.caption(f"LLM: {os.environ.get('LLM_MODEL')}" if has_key else
                   "No LLM configured: agents use the reviewed offline baselines.")
    state = pipeline.load_state(group)
    st.title("Gate console")
    st.markdown(f"<p class='eyebrow'>{style.esc(group)} · <span class='mono'>"
                f"{style.esc(state['model_id'] or 'not discovered yet')}</span></p>", unsafe_allow_html=True)
    progress_row(state)
    kind, item = pipeline.next_action(state)
    if state["last_error"] and kind == "run":
        st.error(f"{STEP_LABELS[state['last_error']['stage']]} failed: {state['last_error']['error']}. "
                 "Fix the cause, then run it again.")
    if kind == "run":
        style.note(f"Next: the {STEP_LABELS[item]} agent. It runs {'with the LLM' if use_llm else 'offline'} and "
                   "stops at the following gate for your decision.")
        if st.button(f"Run {STEP_LABELS[item]}", type="primary"):
            with st.spinner(f"Running {STEP_LABELS[item]}. LLM stages can take several minutes."):
                result = pipeline.run(group, use_llm, progress=lambda message: None)
            if result[0] == "error":
                st.error(pipeline.load_state(group)["last_error"]["error"])
            else:
                st.rerun()
    elif kind == "await":
        st.header(f"{STEP_LABELS[item]}: {pipeline.GATE_NAMES[item].lower()}")
        {"gate1": show_gate1, "gate2": show_gate2, "gate3": show_gate3, "gate4": show_gate4}[item](state)
        review_form(group, item)
    else:
        last_gate = state["steps"].get("gate4") or state["steps"].get("gate2", {})
        outcome = "validated for research" if item == "validated_for_research" else "documented"
        st.success(f"Complete: {outcome}. Final decision by {last_gate.get('reviewer')} on "
                   f"{str(last_gate.get('time'))[:16].replace('T', ' ')} UTC.")
        if "validate" in state["steps"]:
            st.caption("These are the figures the gate 4 reviewer approved.")
            show_gate4(state)
    with st.expander("Audit trail for this model"):
        st.dataframe(pd.DataFrame(state["history"]), hide_index=True, width="stretch")


main()
