"""Group fairness with fairlearn (design doc 4.3): rates per group plus parity metrics, decided by code.

"Selected" means flagged as high risk (score >= decision threshold), the unfavourable outcome. Disparate impact is
computed on the favourable outcome (approved = not flagged), as the four-fifths rule is applied in lending.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, count, demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference,
    false_positive_rate, selection_rate, true_positive_rate,
)


def attributes(policy: dict) -> list[dict]:
    if policy.get("fairness_attributes"):
        return policy["fairness_attributes"]
    return [{"column": policy["fairness_column"], "split": policy["fairness_split"]}]


def group_labels(values: pd.Series, spec: dict) -> pd.Series:
    if "split" in spec:
        numbers = pd.to_numeric(values, errors="coerce")
        labels = np.where(numbers < spec["split"], f"{spec['column']} < {spec['split']}",
                          f"{spec['column']} >= {spec['split']}")
        return pd.Series(labels, index=values.index).where(numbers.notna())
    return values.astype("string").str.strip().str.lower().where(values.notna())


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def fairness_audit(raw: pd.DataFrame, row_ids, y: np.ndarray, scores: np.ndarray, policy: dict):
    """Returns (groups, per-attribute summary, findings, unknowns)."""
    groups, summary, findings, unknowns = [], {}, [], []
    flagged = (scores >= policy["decision_threshold"]).astype(int)
    for spec in attributes(policy):
        column = spec["column"]
        if column not in raw:
            findings.append({"check": f"{column}_group_fairness", "result": "unverifiable", "severity": "medium",
                             "reason": f"Sensitive attribute '{column}' is absent from this dataset"})
            continue
        labels = group_labels(raw.loc[row_ids, column].reset_index(drop=True), spec)
        known = labels.notna().to_numpy()
        if (~known).any():
            unknowns.append(f"{int((~known).sum())} rows have no '{column}' value and are excluded from its groups.")
        yk, fk, lk = y[known].astype(int), flagged[known], labels[known].to_numpy()
        frame = MetricFrame(metrics={"selection_rate": selection_rate, "tpr": true_positive_rate,
                                     "fpr": false_positive_rate, "rows": count},
                            y_true=yk, y_pred=fk, sensitive_features=lk)
        sufficient_all = True
        for name, row in frame.by_group.iterrows():
            events = int(yk[lk == name].sum())
            rows = int(row["rows"])
            sufficient = rows >= policy["min_group_size"] and min(events, rows - events) >= policy["min_group_class_size"]
            sufficient_all &= sufficient
            groups.append({"attribute": column, "group": str(name), "rows": rows, "events": events,
                           "sufficient": sufficient, "selection_rate": _finite(row["selection_rate"]),
                           "approval_rate": _finite(1 - row["selection_rate"]),
                           "tpr": _finite(row["tpr"]), "fpr": _finite(row["fpr"])})
        if len(frame.by_group) < 2 or not sufficient_all:
            findings.append({"check": f"{column}_group_tpr_gap", "result": "unverifiable", "severity": "medium",
                             "reason": "Fewer than two groups, or a group below the minimum size or class support"})
            continue
        metrics = {
            "group_tpr_gap": _finite(frame.difference()["tpr"]),
            "demographic_parity_difference": _finite(demographic_parity_difference(yk, fk, sensitive_features=lk)),
            "disparate_impact": _finite(demographic_parity_ratio(yk, 1 - fk, sensitive_features=lk)),
            "equalized_odds_difference": _finite(equalized_odds_difference(yk, fk, sensitive_features=lk)),
        }
        summary[column] = metrics
        limits = [("group_tpr_gap", "max_tpr_gap", "<="), ("demographic_parity_difference",
                   "max_demographic_parity_difference", "<="), ("disparate_impact", "min_disparate_impact", ">="),
                  ("equalized_odds_difference", "max_equalized_odds_difference", "<=")]
        for metric, key, op in limits:
            value, limit = metrics[metric], policy[key]
            if value is None:
                findings.append({"check": f"{column}_{metric}", "result": "unverifiable", "severity": "medium",
                                 "reason": "Metric undefined (e.g. no favourable outcomes in a group)"})
                continue
            passed = value <= limit if op == "<=" else value >= limit
            findings.append({"check": f"{column}_{metric}", "result": "pass" if passed else "fail",
                             "severity": "medium", "value": value, "limit": limit})
    return groups, summary, findings, unknowns
