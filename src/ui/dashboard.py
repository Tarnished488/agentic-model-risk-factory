"""Portfolio dashboard: pipeline progress, validation results, parity and governance, from aggregate files only."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src import pipeline

RAG_COLOURS = {"green": "#2e7d32", "amber": "#f9a825", "red": "#c62828"}
STEP_ICONS = {"done": "✅", "rejected": "↩️", "next": "▶️", "waiting": "🧑‍⚖️", "error": "⚠️", "": "⬜", "n/a": "—"}


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
                steps[step] = "done"
            elif item == step:
                steps[step] = {"run": "next", "await": "waiting", "error": "error"}[kind]
            else:
                steps[step] = ""
        validation = _read(state["steps"].get("validate", {}).get("state_path"))
        report = _read(validation["metrics_path"]) if validation else None
        rows.append({"group": group, "state": state, "kind": kind, "item": item, "steps": steps, "report": report,
                     "redevelop": _read(state["steps"].get("redevelop", {}).get("state_path"))})
    return rows


def kpis(rows: list[dict], registry: dict) -> None:
    approved = lambda gate: sum(1 for r in rows if r["state"]["steps"].get(gate, {}).get("decision") == "approve")
    waiting = sum(1 for r in rows if r["kind"] == "await")
    columns = st.columns(5)
    columns[0].metric("Models in scope", len(rows))
    columns[1].metric("Documented (gate 2)", approved("gate2"))
    columns[2].metric("Rebuilt in Python (gate 3)", approved("gate3"))
    columns[3].metric("Validated (gate 4)", approved("gate4"))
    columns[4].metric("Waiting for a human", waiting)
    stages = [entry["stage"] for entry in registry["models"].values()]
    if stages:
        st.caption("Registry: " + ", ".join(f"{s} × {stages.count(s)}" for s in sorted(set(stages))))


def progress_board(rows: list[dict]) -> None:
    table = []
    for row in rows:
        record = {"Model": row["group"], "Model ID": row["state"]["model_id"] or "-"}
        record.update({pipeline.GATE_NAMES.get(s, s.title()).replace(" review", " ✓") if s.startswith("gate")
                       else s.title(): STEP_ICONS[row["steps"][s]] for s in pipeline.ORDER})
        record["Next"] = {"run": f"run {row['item']}", "await": f"decision at {row['item']}",
                          "done": row["item"], "error": f"failed at {row['item']}"}[row["kind"]]
        record["Rating"] = (row["report"] or {}).get("overall_rating", "-")
        table.append(record)
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("✅ done · ▶️ next agent · 🧑‍⚖️ waiting for a human · ↩️ rejected, rerunning · ⚠️ failed · — not in scope")


def rag_tiles(report: dict) -> None:
    counts = pd.Series([f.get("rating", "") for f in report["findings"]]).value_counts()
    overall = report.get("overall_rating", "n/a")
    columns = st.columns(4)
    columns[0].markdown(f"<div style='background:{RAG_COLOURS.get(overall, '#607d8b')};color:white;padding:14px;"
                        f"border-radius:10px;text-align:center'><b>Overall</b><br><span style='font-size:28px'>"
                        f"{overall.upper()}</span></div>", unsafe_allow_html=True)
    for column, rating in zip(columns[1:], ("green", "amber", "red")):
        column.markdown(f"<div style='border:3px solid {RAG_COLOURS[rating]};padding:10px;border-radius:10px;"
                        f"text-align:center'><b>{rating.title()}</b><br><span style='font-size:28px'>"
                        f"{int(counts.get(rating, 0))}</span></div>", unsafe_allow_html=True)


def findings_table(report: dict) -> None:
    text = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
    frame = pd.DataFrame([{"Rating": f.get("rating", ""), "Check": f["check"], "Result": f["result"],
                           "Value": text(f.get("value")), "Limit": text(f.get("limit", f.get("reason", ""))),
                           "Recommendation": f.get("recommendation", "")} for f in report["findings"]])
    colour = lambda value: f"background-color:{RAG_COLOURS.get(value, 'white')};color:white" if value in RAG_COLOURS else ""
    st.dataframe(frame.style.map(colour, subset=["Rating"]), hide_index=True, width="stretch")


def charts(report: dict) -> None:
    metrics, curves, policy = report["metrics"], report.get("curves"), report["policy"]
    left, right = st.columns(2)
    if curves:
        roc = pd.DataFrame(curves["roc"])
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="grey")
        left.altair_chart((alt.Chart(roc).mark_line(color="#1565c0").encode(
            x=alt.X("fpr", title="False positive rate"), y=alt.Y("tpr", title="True positive rate"),
            tooltip=["fpr", "tpr"]) + diagonal.encode(x="x", y="y")).properties(
            title=f"ROC (AUC {metrics['auc']:.3f})", height=300), width="stretch")
    calibration = pd.DataFrame(metrics["calibration"])
    perfect = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="grey")
    base = alt.Chart(calibration).encode(x=alt.X("mean_probability", title="Mean predicted"),
                                         y=alt.Y("observed_rate", title="Observed rate"))
    right.altair_chart((base.mark_line(color="#6a1b9a") + base.mark_circle(color="#6a1b9a", opacity=0.8).encode(
        size=alt.Size("count", title="Rows per bin", scale=alt.Scale(type="log", range=[30, 400])),
        tooltip=["bin", "count", "mean_probability", "observed_rate"]) + perfect.encode(x="x", y="y")).properties(
        title=f"Calibration (ECE {metrics['ece']:.4f})", height=300), width="stretch")
    left, right = st.columns(2)
    if curves:
        edges = curves["histogram_edges"]
        good, bad = curves["histogram_good"], curves["histogram_bad"]
        histogram = pd.DataFrame([{"score": (a + b) / 2, "share": g / max(sum(good), 1), "outcome": "Non-default"}
                                  for a, b, g in zip(edges, edges[1:], good)]
                                 + [{"score": (a + b) / 2, "share": d / max(sum(bad), 1), "outcome": "Default"}
                                    for a, b, d in zip(edges, edges[1:], bad)])
        threshold = alt.Chart(pd.DataFrame({"t": [policy["decision_threshold"]]})).mark_rule(strokeDash=[2, 2])
        left.altair_chart((alt.Chart(histogram).mark_bar(opacity=0.6).encode(
            x=alt.X("score:Q", bin=alt.Bin(step=0.05), title="Predicted probability"),
            y=alt.Y("share:Q", stack=None, title="Share of class"), color="outcome:N",
            tooltip=["outcome", "score", "share"]) + threshold.encode(x="t")).properties(
            title="Score distribution by outcome", height=280), width="stretch")
    psi = pd.DataFrame([{"feature": k, "psi": v} for k, v in report["train_holdout_psi"].items() if v is not None])
    if not psi.empty:
        right.altair_chart(alt.Chart(psi).mark_bar(color="#00838f").encode(
            x=alt.X("psi", title="PSI"), y=alt.Y("feature", sort="-x", title=None), tooltip=["feature", "psi"]
        ).properties(title=f"Train/holdout PSI (amber {policy.get('amber', {}).get('train_holdout_psi', 0.1)}, "
                           f"red {policy['max_feature_psi']})", height=280), width="stretch")
    groups = pd.DataFrame(report["groups"])
    if not groups.empty:
        long = groups.melt(id_vars=["group"], value_vars=[c for c in ("approval_rate", "tpr", "fpr") if c in groups],
                           var_name="metric", value_name="rate")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("metric:N", title=None), y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1])),
            color="metric:N", column=alt.Column("group:N", title=None), tooltip=["group", "metric", "rate"]
        ).properties(title=f"Group rates at threshold {policy['decision_threshold']}", width=180, height=220))
        if report.get("fairness"):
            st.dataframe(pd.DataFrame(report["fairness"]).T.rename_axis("attribute"), width="stretch")


def parity_panel(rows: list[dict]) -> None:
    rebuilt = [r for r in rows if r["redevelop"]]
    if not rebuilt:
        st.info("No model has been rebuilt yet.")
        return
    for row in rebuilt:
        history = (_read(row["redevelop"]["parity_result_path"]) or {}).get("history", [])
        frame = pd.DataFrame([{"round": h["round"], "passed": h["passed"],
                               "max prediction diff": (h.get("metrics") or {}).get("max_abs_diff"),
                               "max coefficient diff": (h.get("metrics") or {}).get("coef_max_abs_diff"),
                               "error": (h.get("error") or "").splitlines()[0] if h.get("error") else ""}
                              for h in history])
        st.write(f"**{row['group']}** · mode `{row['redevelop']['mode']}` · retries {row['redevelop']['retry_count']}"
                 f" · IR differences from baseline: {len(row['redevelop']['ir_differences_from_baseline'] or [])}")
        st.dataframe(frame, hide_index=True, width="stretch")


def governance(rows: list[dict], registry: dict) -> None:
    events = [{"model": r["group"], **e} for r in rows for e in r["state"]["history"]]
    if events:
        frame = pd.DataFrame(events)
        frame["time"] = pd.to_datetime(frame["time"])
        frame["what"] = frame.apply(lambda e: e.get("gate") if e["event"] == "gate_decision" else e.get("stage"),
                                    axis=1)
        st.altair_chart(alt.Chart(frame).mark_point(filled=True, size=120).encode(
            x=alt.X("time:T", title=None), y=alt.Y("model:N", title=None),
            color=alt.Color("event:N"), shape=alt.Shape("event:N"),
            tooltip=["model", "event", "what", "decision", "reviewer", "time"]).properties(
            title="Agent runs and human gate decisions", height=60 + 40 * frame["model"].nunique()),
            width="stretch")
        decisions = frame[frame["event"] == "gate_decision"][["time", "model", "gate", "decision", "reviewer"]]
        st.dataframe(decisions.sort_values("time", ascending=False), hide_index=True, width="stretch")
    if registry["models"]:
        st.subheader("Model registry")
        st.dataframe(pd.DataFrame([{"model_id": k, "group": v["group"], "stage": v["stage"],
                                    "version": v["version"][:12], "updated": v["updated_at"]}
                                   for k, v in registry["models"].items()]), hide_index=True, width="stretch")


def render() -> None:
    rows, registry = portfolio(), pipeline.load_registry()
    st.title("Model Risk Dashboard")
    st.caption("Aggregate results only; no borrower-level data is shown. Research validation, not deployment approval.")
    kpis(rows, registry)
    progress_board(rows)
    validated = [r for r in rows if r["report"]]
    tab_validation, tab_parity, tab_governance = st.tabs(["Validation", "Parity", "Governance"])
    with tab_validation:
        if not validated:
            st.info("No validation results yet. Run a model through gate 3 and Validate in the gate console.")
        else:
            row = next(r for r in validated if r["group"] == st.selectbox(
                "Validated model", [r["group"] for r in validated]))
            report = row["report"]
            rag_tiles(report)
            st.markdown(f"**Explainer ({report['explainer_mode']}):** {report['explanation']['summary']}")
            charts(report)
            findings_table(report)
            validation = _read(row["state"]["steps"]["validate"]["state_path"])
            if validation and Path(validation.get("docx_path", "")).is_file():
                st.download_button("Download Word validation report", Path(validation["docx_path"]).read_bytes(),
                                   file_name=f"{report['model_id']}_validation.docx")
    with tab_parity:
        parity_panel(rows)
    with tab_governance:
        governance(rows, registry)
