"""Parity check: compare rebuilt Python predictions with the legacy reference. Pure code, no LLM."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from src.discover.discover_agent import atomic_json, audit, now, output_lock

# Agreed before any rebuild runs, so the LLM cannot tune the bar it is judged against.
THRESHOLDS = {
    # L1: the legacy model file (e.g. LightGBM model.txt) scored in Python on the same rows.
    "inference": {"max_abs_diff": ("<=", 1e-6)},
    # L2: the model re-estimated in Python on the same split.
    "reestimation": {"auc_diff": ("<=", 0.005), "ks_diff": ("<=", 0.01), "spearman": (">=", 0.99)},
}
MAX_RETRIES = 3
COEFFICIENT_LIMIT = 1e-6  # applied on top of any tier whenever coefficients are supplied (GLMs)


def load_predictions(path: Path) -> tuple[dict[str, float], dict[str, int] | None]:
    """Read a CSV with columns row_id, probability and optional label (0/1)."""
    probabilities, labels = {}, {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"row_id", "probability"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path} needs row_id and probability columns")
        for row in reader:
            key = row["row_id"]
            if key in probabilities:
                raise ValueError(f"Duplicate row_id {key} in {path}")
            probabilities[key] = float(row["probability"])
            if row.get("label") not in (None, ""):
                labels[key] = int(row["label"])
    return probabilities, labels or None


def load_coefficients(path: Path) -> dict[str, float]:
    """Read a CSV with columns term and Estimate (R's summary(glm)$coefficients layout)."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"term", "Estimate"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path} needs term and Estimate columns")
        return {row["term"]: float(row["Estimate"]) for row in reader}


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):  # ties share their average rank
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUC needs both classes in the labels")
    rank_sum = sum(r for r, y in zip(_ranks(scores), labels) if y)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def ks(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    pairs = sorted(zip(scores, labels), reverse=True)
    best = tp = fp = 0
    for index, (score, label) in enumerate(pairs):
        tp, fp = tp + label, fp + 1 - label
        if index + 1 == len(pairs) or pairs[index + 1][0] != score:  # evaluate only between tie groups
            best = max(best, abs(tp / positives - fp / negatives))
    return best


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    mean = (len(ra) + 1) / 2
    covariance = sum((x - mean) * (y - mean) for x, y in zip(ra, rb))
    spread = math.sqrt(sum((x - mean) ** 2 for x in ra) * sum((y - mean) ** 2 for y in rb))
    return covariance / spread if spread else float(ra == rb)


def compare(reference: dict[str, float], candidate: dict[str, float],
            labels: dict[str, int] | None, tier: str,
            coefficients: tuple[dict[str, float], dict[str, float]] | None = None) -> dict:
    if tier not in THRESHOLDS:
        raise ValueError(f"Unknown tier {tier}; use one of {sorted(THRESHOLDS)}")
    if reference.keys() != candidate.keys():
        raise ValueError(f"Row ids differ: {len(reference.keys() - candidate.keys())} missing from candidate, "
                         f"{len(candidate.keys() - reference.keys())} extra")
    ids = sorted(reference)
    ref, new = [reference[i] for i in ids], [candidate[i] for i in ids]
    if not ids or not all(math.isfinite(p) and 0 <= p <= 1 for p in ref + new):
        raise ValueError("Predictions must be non-empty probabilities in [0, 1]")
    diffs = [abs(a - b) for a, b in zip(ref, new)]
    metrics = {"rows": len(ids), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(ids),
               "spearman": spearman(ref, new)}
    if labels is not None:
        if not set(ids) <= labels.keys() or not {labels[i] for i in ids} <= {0, 1}:
            raise ValueError("Labels must be 0/1 and cover every compared row")
        y = [labels[i] for i in ids]
        metrics.update(auc_reference=auc(ref, y), auc_candidate=auc(new, y),
                       ks_reference=ks(ref, y), ks_candidate=ks(new, y))
        metrics.update(auc_diff=abs(metrics["auc_reference"] - metrics["auc_candidate"]),
                       ks_diff=abs(metrics["ks_reference"] - metrics["ks_candidate"]))
    checks = []
    for metric, (op, limit) in THRESHOLDS[tier].items():
        if metric not in metrics:
            raise ValueError(f"Tier {tier} needs labels to compute {metric}")
        value = metrics[metric]
        checks.append({"metric": metric, "value": value, "op": op, "limit": limit,
                       "passed": value <= limit if op == "<=" else value >= limit})
    if coefficients is not None:
        ref_coef, new_coef = coefficients
        if ref_coef.keys() != new_coef.keys():
            raise ValueError(f"Coefficient terms differ: {sorted(ref_coef.keys() ^ new_coef.keys())}")
        metrics["coefficient_diffs"] = {term: abs(ref_coef[term] - new_coef[term]) for term in ref_coef}
        metrics["coef_max_abs_diff"] = max(metrics["coefficient_diffs"].values())
        checks.append({"metric": "coef_max_abs_diff", "value": metrics["coef_max_abs_diff"], "op": "<=",
                       "limit": COEFFICIENT_LIMIT, "passed": metrics["coef_max_abs_diff"] <= COEFFICIENT_LIMIT})
    worst = sorted(range(len(ids)), key=diffs.__getitem__, reverse=True)[:5]
    return {"tier": tier, "passed": all(c["passed"] for c in checks), "metrics": metrics, "checks": checks,
            # Feedback for the code-generation retry: where the rebuild disagrees most.
            "worst_rows": [{"row_id": ids[i], "reference": ref[i], "candidate": new[i], "diff": diffs[i]}
                           for i in worst]}


def route(history: list[dict]) -> str:
    if history[-1]["passed"]:
        return "pending_redevelop_review"  # human gate 3
    return "retry" if len(history) <= MAX_RETRIES else "needs_attention"


def record_round(output: Path, model_id: str, result: dict) -> dict:
    """Append one parity round to <output>/<model_id>/parity_result.json and route the workflow."""
    directory = output / model_id
    with output_lock(directory):
        path = directory / "parity_result.json"
        history = json.loads(path.read_text(encoding="utf-8"))["history"] if path.exists() else []
        history.append({"round": len(history) + 1, "time": now(), **result})
        payload = {"model_id": model_id, "status": route(history), "retry_count": len(history) - 1,
                   "history": history}
        atomic_json(path, payload)
        audit(directory, {"node": "redevelop", "event": "parity_round", "model_id": model_id,
                          "round": len(history), "tier": result["tier"], "passed": result["passed"],
                          "status": payload["status"], "output_path": str(path)})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Legacy predictions CSV: row_id,probability[,label]")
    parser.add_argument("candidate", type=Path, help="Python predictions CSV: row_id,probability")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tier", choices=sorted(THRESHOLDS), default="inference")
    parser.add_argument("--output", type=Path, default=Path("artifacts/redevelop"))
    parser.add_argument("--coefficients", type=Path, nargs=2, metavar=("REFERENCE", "CANDIDATE"),
                        help="Optional term,Estimate CSVs for coefficient parity (GLMs)")
    args = parser.parse_args()
    try:
        reference, labels = load_predictions(args.reference)
        candidate, _ = load_predictions(args.candidate)
        coefficients = tuple(map(load_coefficients, args.coefficients)) if args.coefficients else None
        payload = record_round(args.output, args.model_id,
                               compare(reference, candidate, labels, args.tier, coefficients))
    except (ValueError, OSError, KeyError) as error:
        print(f"Parity error: {error}", file=sys.stderr)
        return 1
    last = payload["history"][-1]
    for check in last["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {check['metric']} = {check['value']:.3g} "
              f"(limit {check['op']} {check['limit']})")
    print(f"Status: {payload['status']}  Result: {args.output / args.model_id / 'parity_result.json'}")
    return 0 if last["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
