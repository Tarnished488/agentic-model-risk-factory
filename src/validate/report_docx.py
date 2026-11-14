"""Validation report as Word (design doc 4.8): template + deterministic tables + checked explanation."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

STATUS_TEXT = {"pass": "Pass", "conditional_pass": "Conditional pass", "fail": "Fail"}
RAG_COLOURS = {"green": RGBColor(0x2E, 0x7D, 0x32), "amber": RGBColor(0xF9, 0xA8, 0x25), "red": RGBColor(0xC6, 0x28, 0x28)}


def _table(document, header: list[str], rows: list[list]) -> None:
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = text
    for row in rows:
        for cell, value in zip(table.add_row().cells, row):
            cell.text = f"{value:.6g}" if isinstance(value, float) else str(value)


def _chart(document, charts: dict, name: str) -> None:
    if charts and charts.get(name) and Path(charts[name]).is_file():
        document.add_picture(charts[name], width=Inches(5))


def write_report(report: dict, explanation: dict, approvals: list[dict], path: Path,
                 charts: dict | None = None) -> Path:
    document = Document()
    document.styles["Normal"].font.size = Pt(10)
    document.add_heading("Independent Model Validation Report", 0)
    document.add_paragraph(f"Model: {report['model_id']}    Generated: {report['generated_at']}")
    rating = report.get("overall_rating", "n/a")
    heading = document.add_paragraph()
    run = heading.add_run(f"Overall rating: {rating.upper()}")
    run.bold = True
    if rating in RAG_COLOURS:
        run.font.color.rgb = RAG_COLOURS[rating]
    document.add_paragraph(f"Rule status: {report['status']}    Explainer conclusion: "
                           f"{STATUS_TEXT[explanation['conclusion']]} ({report['explainer_mode']})")
    document.add_paragraph("Research validation only. This report is not deployment approval or regulatory "
                           "certification. Numbers in the narrative are checked against the metrics.").italic = True

    document.add_heading("1. Summary", 1)
    document.add_paragraph(explanation["summary"])
    document.add_paragraph(f"Conclusion reason: {explanation['conclusion_reason']}")

    document.add_heading("2. Performance", 1)
    metrics = report["metrics"]
    _table(document, ["Metric", "Value"], [[k, v] for k, v in metrics.items() if k != "calibration"])
    _chart(document, charts, "roc")
    _chart(document, charts, "score_distribution")
    document.add_paragraph(explanation["performance"])

    document.add_heading("3. Calibration", 1)
    _table(document, ["Bin", "Rows", "Mean predicted", "Observed rate"],
           [[b["bin"], b["count"], b["mean_probability"], b["observed_rate"]] for b in metrics["calibration"]])
    _chart(document, charts, "calibration")
    document.add_paragraph(explanation["calibration"])

    document.add_heading("4. Stability", 1)
    _table(document, ["Feature", "Train/holdout PSI"],
           [[k, v if v is not None else "n/a"] for k, v in report["train_holdout_psi"].items()])
    _chart(document, charts, "psi")
    document.add_paragraph(explanation["stability"])

    document.add_heading("5. Fairness", 1)
    _table(document, ["Group", "Rows", "Events", "TPR", "FPR", "Flag rate", "Approval rate", "Sufficient"],
           [[g["group"], g["rows"], g["events"], g["tpr"], g["fpr"], g["selection_rate"],
             g.get("approval_rate", ""), g["sufficient"]] for g in report["groups"]])
    if report.get("fairness"):
        _table(document, ["Attribute", "TPR gap", "Demographic parity diff.", "Disparate impact (approval)",
                          "Equalized odds diff."],
               [[a, m["group_tpr_gap"], m["demographic_parity_difference"], m["disparate_impact"],
                 m["equalized_odds_difference"]] for a, m in report["fairness"].items()])
    _chart(document, charts, "fairness")
    document.add_paragraph(f"Decision threshold: {report['policy']['decision_threshold']} (fairlearn metrics; "
                           "disparate impact uses the approval rate, four-fifths rule = 0.8)")
    document.add_paragraph(explanation["fairness"])

    document.add_heading("6. Conceptual soundness", 1)
    if report["coefficients"]:
        _table(document, ["Feature", "Coefficient"], [[k, v] for k, v in report["coefficients"].items()])
    document.add_paragraph(explanation["conceptual_soundness"])

    document.add_heading("7. Rule checks (red / amber / green)", 1)
    _table(document, ["Check", "Rating", "Result", "Value", "Limit", "Recommendation"],
           [[f["check"], f.get("rating", "").upper(), f["result"], f.get("value", ""),
             f.get("limit", f.get("reason", "")), f.get("recommendation", "")] for f in report["findings"]])

    document.add_heading("8. Limitations and recommendations", 1)
    for item in explanation["limitations"]:
        document.add_paragraph(item, style="List Bullet")
    for item in explanation["recommendations"]:
        document.add_paragraph("Recommendation: " + item, style="List Bullet")

    document.add_heading("9. Governance trail", 1)
    _table(document, ["Gate", "Reviewer", "Decision", "Time"],
           [[a.get("gate", "discover"), a.get("reviewer", ""), a.get("decision", ""), a.get("time", "")]
            for a in approvals])
    document.add_paragraph(f"Policy {report['policy']['version']}: {report['policy']['description']}")
    document.save(path)
    return path
