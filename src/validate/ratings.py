"""Red / amber / green per finding (design doc 4.5) and a recommendation for each, decided by code."""
from __future__ import annotations

# (substring in the check name, amber key, direction): "min" means lower values are worse.
FAMILIES = [("holdout_auc", "holdout_auc", "min"), ("holdout_ks", "holdout_ks", "min"),
            ("calibration_ece", "calibration_ece", "max"), ("train_holdout_psi", "train_holdout_psi", "max"),
            ("group_tpr_gap", "group_tpr_gap", "max"),
            ("demographic_parity_difference", "demographic_parity_difference", "max"),
            ("disparate_impact", "disparate_impact", "min"),
            ("equalized_odds_difference", "equalized_odds_difference", "max")]

RECOMMENDATIONS = {
    "holdout_auc": "Investigate discriminatory power; consider additional predictors or redevelopment.",
    "holdout_ks": "Investigate separation between goods and bads; review the score scale and segmentation.",
    "calibration_ece": "Recalibrate on a holdout (e.g. isotonic or Platt) and review sparse high-score bins.",
    "train_holdout_psi": "Investigate the distribution shift of this feature before relying on the model.",
    "group_tpr_gap": "Review error-rate parity across groups; document business justification or mitigate.",
    "demographic_parity_difference": "Review flag-rate differences across groups; document justification.",
    "disparate_impact": "Approval-rate ratio is near or below the four-fifths rule; justify or mitigate.",
    "equalized_odds_difference": "Review TPR/FPR differences across groups; consider threshold or model changes.",
    "coefficient_sign": "Review this feature's direction with the business; a wrong sign suggests misspecification.",
    "score_direction": "AUC is below 0.5, so scores likely point at the wrong class: check the probability column and event level (1 - AUC is the corrected value) before reading other findings.",
}


def family(check: str) -> str | None:
    if check.startswith("coefficient_sign"):
        return "coefficient_sign"
    return next((name for name, _, _ in FAMILIES if name in check), check if check in RECOMMENDATIONS else None)


def rate(finding: dict, amber: dict) -> str:
    if finding["result"] == "fail":
        return "red"
    if finding["result"] == "unverifiable":
        return "amber"
    for name, key, direction in FAMILIES:
        if name in finding["check"] and key in amber and isinstance(finding.get("value"), (int, float)):
            value, band = finding["value"], amber[key]
            if (direction == "min" and value < band) or (direction == "max" and value > band):
                return "amber"
    return "green"


def annotate(findings: list[dict], policy: dict) -> str:
    """Adds rating and recommendation to every finding; returns the overall rating."""
    amber = policy.get("amber", {})
    for finding in findings:
        finding["rating"] = rate(finding, amber)
        name = family(finding["check"])
        if finding["rating"] == "green":
            finding["recommendation"] = "No action; keep under periodic monitoring."
        elif finding["result"] == "unverifiable":
            finding["recommendation"] = "Supply the missing input or document why this check cannot apply."
        else:
            finding["recommendation"] = RECOMMENDATIONS.get(name, "Review with the model owner.")
    ratings = {f["rating"] for f in findings}
    return "red" if "red" in ratings else "amber" if "amber" in ratings else "green"
