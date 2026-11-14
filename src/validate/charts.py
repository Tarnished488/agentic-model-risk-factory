"""Validation charts as PNG (design doc: chart images), drawn only from aggregate report data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display; safe in subprocesses and servers
import matplotlib.pyplot as plt  # noqa: E402

RAG = {"green": "#2e7d32", "amber": "#f9a825", "red": "#c62828"}


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def draw_charts(report: dict, directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths, metrics, curves = {}, report["metrics"], report["curves"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([p["fpr"] for p in curves["roc"]], [p["tpr"] for p in curves["roc"]], color="#1565c0",
            label=f"AUC {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (holdout)")
    ax.legend(loc="lower right")
    paths["roc"] = str(_save(fig, directory / "roc.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    bins = metrics["calibration"]
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect calibration")
    ax.plot([b["mean_probability"] for b in bins], [b["observed_rate"] for b in bins], "o-", color="#6a1b9a",
            label=f"Model (ECE {metrics['ece']:.4f})")
    for b in bins:
        ax.annotate(str(b["count"]), (b["mean_probability"], b["observed_rate"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -10))
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Calibration (rows per bin)")
    ax.legend(loc="upper left")
    paths["calibration"] = str(_save(fig, directory / "calibration.png"))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    edges = curves["histogram_edges"]
    centres, width = [(a + b) / 2 for a, b in zip(edges, edges[1:])], edges[1] - edges[0]
    good, bad = curves["histogram_good"], curves["histogram_bad"]
    ax.bar(centres, [g / max(sum(good), 1) for g in good], width=width, alpha=0.6, label="Non-default")
    ax.bar(centres, [b / max(sum(bad), 1) for b in bad], width=width, alpha=0.6, label="Default")
    ax.axvline(report["policy"]["decision_threshold"], color="black", linestyle=":", label="Decision threshold")
    ax.set(xlabel="Predicted probability", ylabel="Share of class", title="Score distribution by outcome")
    ax.legend()
    paths["score_distribution"] = str(_save(fig, directory / "score_distribution.png"))

    groups = [g for g in report["groups"] if g["tpr"] is not None]
    if groups:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        names, width = [g["group"] for g in groups], 0.25
        for offset, (key, label) in enumerate([("approval_rate", "Approval rate"), ("tpr", "TPR"),
                                               ("fpr", "FPR")]):
            ax.bar([i + (offset - 1) * width for i in range(len(groups))],
                   [g.get(key) or 0 for g in groups], width=width, label=label)
        ax.set_xticks(range(len(groups)), names)
        ax.set(ylim=(0, 1), title=f"Group rates at threshold {report['policy']['decision_threshold']}")
        ax.legend()
        paths["fairness"] = str(_save(fig, directory / "fairness.png"))

    psi = {k: v for k, v in report["train_holdout_psi"].items() if v is not None}
    if psi:
        fig, ax = plt.subplots(figsize=(5.5, 0.5 + 0.45 * len(psi)))
        ax.barh(list(psi), list(psi.values()), color="#00838f")
        ax.axvline(report["policy"].get("amber", {}).get("train_holdout_psi", 0.1), color=RAG["amber"],
                   linestyle="--", label="Amber")
        ax.axvline(report["policy"]["max_feature_psi"], color=RAG["red"], linestyle="--", label="Red")
        ax.set(xlabel="PSI", title="Train / holdout stability")
        ax.legend(loc="lower right")
        paths["psi"] = str(_save(fig, directory / "psi.png"))
    return paths


def log_validation_mlflow(run_dir: Path, report: dict, files: list[Path]) -> None:
    """One MLflow run per validation (needs mlflow-skinny); failures are handled by the caller."""
    import mlflow

    with mlflow.start_run(run_name="validate-" + run_dir.name):
        mlflow.set_tags({"node": "validate", "model_id": report["model_id"], "status": report["status"],
                         "overall_rating": report["overall_rating"], "explainer": report["explainer_mode"]})
        mlflow.log_params({"policy_version": report["policy"]["version"],
                           "decision_threshold": report["policy"]["decision_threshold"]})
        mlflow.log_metrics({k: v for k, v in report["metrics"].items() if isinstance(v, (int, float))})
        for attribute, values in (report.get("fairness") or {}).items():
            mlflow.log_metrics({f"{attribute}_{k}": v for k, v in values.items() if v is not None})
        for path in files:
            if path.exists():
                mlflow.log_artifact(str(path))
