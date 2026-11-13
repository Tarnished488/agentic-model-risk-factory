"""Hand-written Python rebuild of the GMSC probit GLM, following gmsc_probit_ir.json.

It is the reference the LLM code generator must match, and the demo fallback if generation fails.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

TARGET = "SeriousDlqin2yrs"
FEATURES = ["RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate", "age"]
# Coefficient names as R prints them, so coefficient parity compares like with like.
R_TERMS = {"const": "(Intercept)", "RevolvingUtilizationOfUnsecuredLines": "unsecured.credit",
           "NumberOfTime30-59DaysPastDueNotWorse": "`<2months Deliquency`",
           "NumberOfTime60-89DaysPastDueNotWorse": "`<3months Deliquency`",
           "NumberOfTimes90DaysLate": "`3MonthsDefault`", "age": "age"}


def r_as_character(value: float) -> str:
    """R's as.character for doubles: 15 significant digits, no trailing '.0'."""
    return f"{value:.15g}"


def load(path: Path) -> pd.DataFrame:
    # pandas 3 parses decimals round-trip exactly by default, matching readr.
    frame = pd.read_csv(path)
    return frame.rename(columns={frame.columns[0]: "row_id"})


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """IR filters then imputation, in the notebook's order."""
    frame = frame[frame["age"] > 0]
    late_as_text = frame["NumberOfTimes90DaysLate"].map(r_as_character)
    frame = frame[(late_as_text < "96") & (frame["DebtRatio"] <= 1)
                  & (frame["RevolvingUtilizationOfUnsecuredLines"] <= 1)].copy()
    frame["NumberOfDependents"] = frame["NumberOfDependents"].fillna(0)
    # Median over all filtered rows before the split, as in the notebook.
    frame["MonthlyIncome"] = frame["MonthlyIncome"].fillna(frame["MonthlyIncome"].median())
    return frame


EPS = np.finfo(float).eps
THRESH = -norm.ppf(EPS)  # R's probit linkinv clamps eta to +/- this


def _probit_mu(eta: np.ndarray) -> np.ndarray:
    return norm.cdf(np.clip(eta, -THRESH, THRESH))


def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
    return float(-2 * np.sum(np.where(y == 1, np.log(mu), np.log1p(-mu))))


def fit_glm_probit_like_r(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-8, maxit: int = 25) -> np.ndarray:
    """Port of R's glm.fit IRLS for binomial(link = probit): same start, step and stop rule.

    A generic optimiser (e.g. statsmodels) converges further than R and differs by ~1e-5.
    """
    eta = norm.ppf((y + 0.5) / 2)  # binomial initialize: mustart = (y + 0.5) / 2
    mu = _probit_mu(eta)
    dev_old = _deviance(y, mu)
    for _ in range(maxit):
        mu_eta = np.maximum(norm.pdf(eta), EPS)
        z = eta + (y - mu) / mu_eta
        w = np.sqrt(mu_eta ** 2 / (mu * (1 - mu)))
        coef = np.linalg.lstsq(x * w[:, None], z * w, rcond=None)[0]
        eta = x @ coef
        mu = _probit_mu(eta)
        dev = _deviance(y, mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < epsilon:
            return coef
        dev_old = dev
    raise ValueError("IRLS did not converge")


def _design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame[FEATURES].to_numpy(dtype=float)])


class Model:
    """Interface required of every rebuild: fit, predict_proba, get_coefficients."""

    def fit(self, frame: pd.DataFrame) -> "Model":
        self.coef = fit_glm_probit_like_r(_design(frame), frame[TARGET].to_numpy(dtype=float))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return _probit_mu(_design(frame) @ self.coef)

    def get_coefficients(self) -> dict[str, float]:
        return {R_TERMS[name]: float(value) for name, value in zip(["const", *FEATURES], self.coef)}


def run(data_dir: Path, reference_dir: Path, output: Path) -> dict:
    frame = prepare(load(data_dir / "cs-training.csv"))
    train_ids = set(pd.read_csv(reference_dir / "train_ids.csv")["row_id"])
    unknown = train_ids - set(frame["row_id"])
    if unknown:
        raise ValueError(f"{len(unknown)} R training rows are missing after Python filtering")
    is_train = frame["row_id"].isin(train_ids)
    model = Model().fit(frame[is_train])
    holdout = frame[~is_train]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": holdout["row_id"], "probability": model.predict_proba(holdout),
                  "label": holdout[TARGET].astype(int)}).to_csv(output / "holdout_predictions.csv", index=False)
    coefficients = model.get_coefficients()
    pd.DataFrame({"term": list(coefficients), "Estimate": list(coefficients.values())}).to_csv(
        output / "coefficients.csv", index=False, float_format="%.17g")
    return {"filtered_rows": len(frame), "train_rows": int(is_train.sum()), "holdout_rows": len(holdout)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/GiveMeSomeCredit"))
    parser.add_argument("--reference", type=Path, default=Path("data/reference/gmsc_probit"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/redevelop/MODEL-9239165C4AE5/candidate"))
    args = parser.parse_args()
    print(run(args.data, args.reference, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
