"""Synthetic-data sandbox for the public demo: no borrower data, no R and no LLM.

Real Give Me Some Credit data and the R reference outputs never leave the team's machines, so a cloud deployment
cannot run Redevelop or Validate on them. The sandbox generates rows with the GMSC column layout, uses the
hand-written Python baseline (which matches R to 1e-14 on the real data) as the parity reference, and points the
pipeline at its own folder so nothing touches the real artifacts. Results describe random rows, not the real model.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.redevelop.baselines import gmsc_probit

ROOT = Path(tempfile.gettempdir()) / "model_risk_factory_sandbox"


def synthetic_gmsc(path: Path, rows: int = 3000, intercept: float = -1.2, slope: float = 1.0) -> None:
    """Random rows in the Give Me Some Credit layout; labels depend on utilisation and 90-day lateness."""
    rng = np.random.default_rng(1)
    late = rng.choice([0, 0, 0, 1, 2, 98], rows)
    frame = pd.DataFrame({
        '': np.arange(1, rows + 1),
        'SeriousDlqin2yrs': 0,
        'RevolvingUtilizationOfUnsecuredLines': rng.uniform(0, 1.2, rows),
        'age': rng.integers(0, 90, rows),
        'NumberOfTime30-59DaysPastDueNotWorse': rng.integers(0, 4, rows),
        'DebtRatio': rng.uniform(0, 1.3, rows),
        'MonthlyIncome': np.where(rng.uniform(size=rows) < 0.2, np.nan, rng.uniform(1000, 9000, rows)),
        'NumberOfOpenCreditLinesAndLoans': rng.integers(0, 10, rows),
        'NumberOfTimes90DaysLate': late,
        'NumberRealEstateLoansOrLines': rng.integers(0, 3, rows),
        'NumberOfTime60-89DaysPastDueNotWorse': rng.integers(0, 3, rows),
        'NumberOfDependents': np.where(rng.uniform(size=rows) < 0.1, np.nan, rng.integers(0, 4, rows)),
    })
    signal = intercept + slope * (1.5 * frame['RevolvingUtilizationOfUnsecuredLines'] + 0.4 * np.minimum(late, 3))
    frame['SeriousDlqin2yrs'] = (rng.uniform(size=rows) < 1 / (1 + np.exp(-signal))).astype(int)
    frame.to_csv(path, index=False)


def prepare(root: Path | None = None) -> Path:
    """Create the synthetic data, baseline reference, policy and pipeline config once; return the config path."""
    root = root or ROOT
    config_path = root / "pipeline.json"
    if config_path.exists():
        return config_path
    data, reference = root / "data", root / "reference"
    data.mkdir(parents=True, exist_ok=True)
    reference.mkdir(exist_ok=True)
    # About 10% defaults and AUC near 0.76: closer to real credit data than the test fixture's defaults.
    synthetic_gmsc(data / "cs-training.csv", rows=20000, intercept=-4.6, slope=2.0)
    kept = gmsc_probit.prepare(gmsc_probit.load(data / "cs-training.csv"))
    pd.DataFrame({"row_id": kept["row_id"].sample(frac=0.7, random_state=0)}).to_csv(
        reference / "train_ids.csv", index=False)
    gmsc_probit.run(data, reference, reference)
    # Synthetic labels ignore age and the shorter past-due counts, so their coefficient signs are random here.
    policy = json.loads(Path("config/validation_policy.json").read_text(encoding="utf-8"))
    (root / "policy.json").write_text(json.dumps({**policy, "expected_signs": {}}), encoding="utf-8")
    config = json.loads(Path("config/pipeline.json").read_text(encoding="utf-8"))
    config["models"]["gmsc_probit"] = {
        **config["models"]["gmsc_probit"], "validation_policy": str(root / "policy.json"),
        "redevelop": {**config["models"]["gmsc_probit"]["redevelop"], "data": str(data), "reference": str(reference)}}
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")  # written last: marks setup complete
    return config_path


def activate(pipeline, root: Path | None = None) -> None:
    """Point the pipeline module at the sandbox config and artifact folder."""
    root = root or ROOT
    pipeline.CONFIG = prepare(root)
    pipeline.ARTIFACTS = root / "artifacts"


def reset(root: Path | None = None) -> None:
    shutil.rmtree(root or ROOT, ignore_errors=True)
