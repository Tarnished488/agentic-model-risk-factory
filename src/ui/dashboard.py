"""Portfolio dashboard: pipeline ledger, validation results, parity and governance, from aggregate files only."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import pipeline
from src.ui.style import BLUE, INK, MUTED, RAG, RULE, esc, kpi_strip, note, rating_band, stamp

STEP_NAMES = ["Discover", "Gate 1", "Document", "Gate 2", "Redevelop", "Gate 3", "Validate", "Gate 4"]
STAMPS = {"approved": "Approved", "done": "Done", "rejected": "Rejected", "waiting": "Your decision",
          "next": "Next", "error": "Failed", "": "", "n/a": "Not in scope"}


def _read(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def portfolio() -> list[dict]:
    rows = []
    for group, settings in pipeline.config()["models"].items():
        state = pipeline.load_state(group)
        kind, item = pipeline.next_action(state)
        if state["last_error"] and kind == "run":
            kind = "error"
        steps = {}
        for step in pipeline.ORDER:
            info = state["steps"].get(step)
            if not settings["redevelop"] and step not in ("discover", "gate1", "document", "gate2"):
                steps[step] = "n/a"
            elif info and info.get("decision") == "reject":
                steps[step] = "rejected"
            elif info:
                steps[step] = "approved" if step.startswith("gate") else "done"
            elif item == step:
                steps[step] = {"run": "next", "await": "waiting", "error": "error"}[kind]
            else:
                steps[step] = ""
        validation = _read(state["steps"].get("validate", {}).get("state_path"))
        report = _read(validation["metrics_path"]) if validation else None
        rows.append({"group": group, "state": state, "kind": kind, "item": item, "steps": steps, "report": report,
                     "validation": validation,
                     "redevelop": _read(state["steps"].get("redevelop", {}).get("state_path"))})
    return rows


def kpis(rows: list[dict]) -> None:
    approved = lambda gate: sum(1 for r in rows if r["state"]["steps"].get(gate, {}).get("decision") == "approve")
    kpi_strip([("models in scope", str(len(rows))), ("documented and approved", str(approved("gate2"))),
               ("rebuilt in Python", str(approved("gate3"))), ("validated for research", str(approved("gate4"))),
               ("waiting for your decision", str(sum(1 for r in rows if r["kind"] == "await")))])


def ledger(rows: list[dict]) -> None:
    head = "".join(f"<th><span class='n'>{i}</span>{name}</th>" for i, name in enumerate(STEP_NAMES, 1))
    body = []
    for row in rows:
        cells = "".join(f"<td>{stamp(row['steps'][s] or 'none', STAMPS[row['steps'][s]] or '·')}</td>"
                        if row['steps'][s] != "n/a" else "<td><span class='stamp none'>–</span></td>"
                        for s in pipeline.ORDER)
        rating = (row["report"] or {}).get("overall_rating")
        rating_cell = (f"<span class='stamp' style='color:{RAG[rating]};border-color:{RAG[rating]}'>"
                       f"{rating.title()}</span>" if rating in RAG else "<span class='stamp none'>–</span>")
        body.append(f"<tr><td class='model'><b>{esc(row['group'])}</b><span style='white-space:nowrap'>"
                    f"{esc(row['state']['model_id'] or 'not discovered')}"
                    f"</span></td>{cells}<td>{rating_cell}</td></tr>")
    st.markdown(f"<div class='ledger-wrap' tabindex='0'><table class='ledger' aria-label='Pipeline ledger'><thead>"
                f"<tr><th>Model</th>{head}<th>Rating</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>",
                unsafe_allow_html=True)
    st.caption("Agents run between the gates; each gate needs a named reviewer's decision before the next agent "
               "starts. Not in scope: the model is documented only, not rebuilt.")


def charts(report: dict) -> None:
    metrics, curves, policy = report["metrics"], report.get("curves"), report["policy"]
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=RULE)
    left, right = st.columns(2, gap="large")
    if curves:
        roc = pd.DataFrame(curves["roc"])
        left.altair_chart((alt.Chart(roc).mark_line(color=BLUE, strokeWidth=2.5).encode(
            x=alt.X("fpr", title="False positive rate"), y=alt.Y("tpr", title="True positive rate"),
            tooltip=[alt.Tooltip("fpr", format=".3f"), alt.Tooltip("tpr", format=".3f")])
            + diagonal.encode(x="x", y="y")).properties(title=f"ROC · AUC {metrics['auc']:.3f}", height=290),
            width="stretch")
    calibration = pd.DataFrame(metrics["calibration"])
    base = alt.Chart(calibration).encode(x=alt.X("mean_probability", title="Mean predicted probability"),
                                         y=alt.Y("observed_rate", title="Observed default rate"))
    right.altair_chart((base.mark_line(color=INK, strokeWidth=1.5) + base.mark_circle(color=INK, opacity=0.85).encode(
        size=alt.Size("count", title="Rows per bin", scale=alt.Scale(type="log", range=[25, 420])),
        tooltip=["bin", "count", alt.Tooltip("mean_probability", format=".3f"),
                 alt.Tooltip("observed_rate", format=".3f")]) + diagonal.encode(x="x", y="y")).properties(
        title=f"Calibration · ECE {metrics['ece']:.4f}", height=290), width="stretch")
    left, right = st.columns(2, gap="large")
    if curves:
        edges, good, bad = curves["histogram_edges"], curves["histogram_good"], curves["histogram_bad"]
        histogram = pd.DataFrame(
            [{"score": (a + b) / 2, "share": g / max(sum(good), 1), "outcome": "Repaid"}
             for a, b, g in zip(edges, edges[1:], good)]
            + [{"score": (a + b) / 2, "share": d / max(sum(bad), 1), "outcome": "Defaulted"}
               for a, b, d in zip(edges, edges[1:], bad)])
        threshold = alt.Chart(pd.DataFrame({"t": [policy["decision_threshold"]]})).mark_rule(
            color=INK, strokeDash=[2, 3])
        left.altair_chart((alt.Chart(histogram).mark_bar(opacity=0.75, binSpacing=1).encode(
            x=alt.X("score:Q", bin=alt.Bin(step=0.05), title="Predicted probability"),
            y=alt.Y("share:Q", stack=None, title="Share of each outcome"),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=["Repaid", "Defaulted"], range=["#7A8FA6", RAG["red"]]),
                            title=None), tooltip=["outcome", alt.Tooltip("share", format=".3f")])
            + threshold.encode(x="t")).properties(
            title=f"Scores by outcome · cut-off {policy['decision_threshold']}", height=270), width="stretch")
    psi = pd.DataFrame([{"feature": k, "psi": v} for k, v in report["train_holdout_psi"].items() if v is not None])
    if not psi.empty:
        amber_line = policy.get("amber", {}).get("train_holdout_psi", 0.1)
        rules = alt.Chart(pd.DataFrame({"x": [amber_line, policy["max_feature_psi"]],
                                        "c": [RAG["amber"], RAG["red"]]})).mark_rule(strokeDash=[3, 3]).encode(
            x="x", color=alt.Color("c:N", scale=None))
        bars = alt.Chart(psi).mark_bar(color=BLUE).encode(
            x=alt.X("psi", title="PSI (dashed: amber and red limits)"), y=alt.Y("feature", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("psi", format=".5f")])
        # Values are often far below the limits, so print them: an invisible bar is itself the finding.
        labels = bars.mark_text(align="left", dx=4, color=INK, fontSize=11).encode(text=alt.Text("psi", format=".4f"))
        right.altair_chart((bars + labels + rules).properties(title="Train / holdout stability", height=270),
                           width="stretch")
    groups = pd.DataFrame(report["groups"])
    if not groups.empty:
        names = {"approval_rate": "Approval rate", "tpr": "True positive rate", "fpr": "False positive rate"}
        long = groups.melt(id_vars=["group"], value_vars=[c for c in names if c in groups],
                           var_name="metric", value_name="rate").replace({"metric": names})
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            y=alt.Y("group:N", title=None), x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color("metric:N", title=None), yOffset="metric:N",
            tooltip=["group", "metric", alt.Tooltip("rate", format=".3f")]).properties(
            title=f"Group outcomes at cut-off {policy['decision_threshold']}", height=60 + 70 * groups["group"].nunique()),
            width="stretch")
        if report.get("fairness"):
            table = pd.DataFrame(report["fairness"]).T.rename_axis("attribute").rename(columns={
                "group_tpr_gap": "TPR gap", "demographic_parity_difference": "Demographic parity diff.",
                "disparate_impact": "Disparate impact (approval)", "equalized_odds_difference": "Equalized odds diff."})
            st.dataframe(table.style.format("{:.3f}"), width="stretch")


def findings_table(report: dict) -> None:
    text = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
    frame = pd.DataFrame([{"Rating": f.get("rating", "").title(), "Check": f["check"], "Result": f["result"],
                           "Value": text(f.get("value")), "Limit": text(f.get("limit", f.get("reason", ""))),
                           "Recommendation": f.get("recommendation", "")} for f in report["findings"]])
    order = {"Red": 0, "Amber": 1, "Green": 2}
    frame = frame.sort_values("Rating", key=lambda s: s.map(order).fillna(3), kind="stable")
    colour = lambda v: f"color:{RAG.get(v.lower(), INK)};font-weight:600" if v else ""
    st.dataframe(frame.style.map(colour, subset=["Rating"]), hide_index=True, width="stretch")


def validation_tab(rows: list[dict]) -> None:
    validated = [r for r in rows if r["report"]]
    if not validated:
        note("No validation results yet. Take a model through gate 3 in the gate console, then run Validate.")
        return
    choice = st.selectbox("Model", [r["group"] for r in validated]) if len(validated) > 1 else validated[0]["group"]
    row = next(r for r in validated if r["group"] == choice)
    report = row["report"]
    counts = pd.Series([f.get("rating", "") for f in report["findings"]]).value_counts().to_dict()
    rating_band(report.get("overall_rating", "n/a"), counts)
    st.markdown(f"<p><b>Explanation</b> <span class='mono' style='color:{MUTED}'>({esc(report['explainer_mode'])})"
                f"</span><br>{esc(report['explanation']['summary'])}</p>", unsafe_allow_html=True)
    charts(report)
    st.subheader("Findings, worst first")
    findings_table(report)
    if row["validation"] and Path(row["validation"].get("docx_path", "")).is_file():
        st.download_button("Download the Word validation report", Path(row["validation"]["docx_path"]).read_bytes(),
                           file_name=f"{report['model_id']}_validation.docx")


def parity_tab(rows: list[dict]) -> None:
    rebuilt = [r for r in rows if r["redevelop"]]
    if not rebuilt:
        note("No model has been rebuilt yet.")
        return
    for row in rebuilt:
        history = (_read(row["redevelop"]["parity_result_path"]) or {}).get("history", [])
        frame = pd.DataFrame([{"Round": h["round"], "Result": "Pass" if h["passed"] else "Fail",
                               "Max prediction diff": (h.get("metrics") or {}).get("max_abs_diff"),
                               "Max coefficient diff": (h.get("metrics") or {}).get("coef_max_abs_diff"),
                               "Error": (h.get("error") or "").splitlines()[0] if h.get("error") else ""}
                              for h in history])
        differences = len(row["redevelop"]["ir_differences_from_baseline"] or [])
        st.markdown(f"**{esc(row['group'])}** · {esc(row['redevelop']['mode'])} rebuild · "
                    f"{row['redevelop']['retry_count']} retries · {differences} IR differences from the reviewed baseline")
        st.caption("Migration scope: " + row['redevelop'].get('migration_scope', 'reestimation'))
        st.dataframe(frame.style.format({"Max prediction diff": "{:.2e}", "Max coefficient diff": "{:.2e}"},
                                        na_rep=""), hide_index=True, width="stretch")


def governance_tab(rows: list[dict], registry: dict) -> None:
    events = [{"model": r["group"], **e} for r in rows for e in r["state"]["history"]]
    if not events:
        note("Nothing has run yet.")
    else:
        frame = pd.DataFrame(events)
        frame["time"] = pd.to_datetime(frame["time"])
        frame["kind"] = frame["event"].map({"gate_decision": "Human decision", "stage_completed": "Agent run",
                                            "stage_failed": "Agent failed"})
        frame["step"] = frame.apply(lambda e: e.get("gate") if e["event"] == "gate_decision" else e.get("stage"), axis=1)
        st.altair_chart(alt.Chart(frame).mark_point(filled=True, size=110).encode(
            x=alt.X("time:T", title=None), y=alt.Y("model:N", title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Agent run", "Human decision", "Agent failed"],
                                                      range=["#7A8FA6", BLUE, RAG["red"]]), title=None),
            shape=alt.Shape("kind:N", title=None),
            tooltip=["model", "kind", "step", "decision", "reviewer", "time"]).properties(
            title="Agent runs and human decisions", height=70 + 45 * frame["model"].nunique()), width="stretch")
        decisions = frame[frame["event"] == "gate_decision"][["time", "model", "gate", "decision", "reviewer"]]
        st.dataframe(decisions.sort_values("time", ascending=False), hide_index=True, width="stretch")
    st.subheader("Model registry")
    if registry["models"]:
        st.dataframe(pd.DataFrame([{"Model ID": k, "Model": v["group"], "Stage": v["stage"],
                                    "Code version": v["version"][:12], "Updated": v["updated_at"][:19]}
                                   for k, v in registry["models"].items()]), hide_index=True, width="stretch")
    else:
        note("Models enter the registry as Staging when gate 3 approves a rebuild.")


def render() -> None:
    rows, registry = portfolio(), pipeline.load_registry()
    st.title("Model risk control ledger")
    st.markdown("<p class='eyebrow'>Discover, document, rebuild and validate legacy credit models, with a named "
                "reviewer at every gate. Aggregate results only; research validation, not deployment approval.</p>",
                unsafe_allow_html=True)
    kpis(rows)
    ledger(rows)
    validation, parity, governance = st.tabs(["Validation", "Parity", "Governance"])
    with validation:
        validation_tab(rows)
    with parity:
        parity_tab(rows)
    with governance:
        governance_tab(rows, registry)
