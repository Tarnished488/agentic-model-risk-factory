"""Hand-written pytest suite for a GMSC probit rebuild, written from the IR only.

The sandbox copies it next to the candidate as test_new_model.py. It is the offline stand-in for the
test-generation agent and the reference that generated tests are compared with.
"""
import numpy as np
import pandas as pd

import new_model

COLUMNS = ["", "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age",
           "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
           "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
           "NumberOfDependents"]
TERMS = ["(Intercept)", "unsecured.credit", "`<2months Deliquency`", "`<3months Deliquency`", "`3MonthsDefault`", "age"]


def synthetic(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "": np.arange(1, rows + 1), "SeriousDlqin2yrs": rng.integers(0, 2, rows),
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, rows), "age": rng.integers(21, 80, rows),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 3, rows), "DebtRatio": rng.uniform(0, 1, rows),
        "MonthlyIncome": rng.uniform(1000, 9000, rows), "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 9, rows),
        "NumberOfTimes90DaysLate": rng.integers(0, 3, rows), "NumberRealEstateLoansOrLines": rng.integers(0, 3, rows),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, rows),
        "NumberOfDependents": rng.integers(0, 4, rows).astype(float)})
    return frame[COLUMNS]


def run_on(tmp_path, frame, train_ids):
    data, reference, out = tmp_path / "data", tmp_path / "reference", tmp_path / "out"
    data.mkdir()
    reference.mkdir()
    frame.to_csv(data / "cs-training.csv", index=False)
    pd.DataFrame({"row_id": train_ids}).to_csv(reference / "train_ids.csv", index=False)
    new_model.run(data, reference, out)
    return pd.read_csv(out / "holdout_predictions.csv"), pd.read_csv(out / "coefficients.csv")


def test_filters_drop_each_excluded_row_and_keep_boundaries(tmp_path):
    frame = synthetic()
    frame.loc[0, "age"] = 0                                         # age > 0
    frame.loc[1, "NumberOfTimes90DaysLate"] = 98                    # 96/98 codes, compared as text
    frame.loc[2, "NumberOfTimes90DaysLate"] = 96
    frame.loc[3, "DebtRatio"] = 1.5                                 # DebtRatio <= 1
    frame.loc[4, "RevolvingUtilizationOfUnsecuredLines"] = 1.01    # utilisation <= 1
    frame.loc[5, "DebtRatio"] = 1.0                                 # boundary is kept
    frame.loc[6, "NumberOfTimes90DaysLate"] = 17                    # "17" < "96" as text: kept
    train_ids = frame[""].iloc[50:].tolist()
    holdout, _ = run_on(tmp_path, frame, train_ids)
    assert sorted(holdout["row_id"]) == [6, 7] + list(range(8, 51))


def test_missing_values_do_not_break_scoring(tmp_path):
    frame = synthetic()
    frame.loc[::5, "MonthlyIncome"] = np.nan
    frame.loc[::7, "NumberOfDependents"] = np.nan
    holdout, _ = run_on(tmp_path, frame, frame[""].iloc[100:].tolist())
    assert len(holdout) == 100
    assert holdout["probability"].between(0, 1).all()


def test_outputs_have_contract_columns_and_terms(tmp_path):
    frame = synthetic()
    holdout, coefficients = run_on(tmp_path, frame, frame[""].iloc[100:].tolist())
    assert list(holdout.columns) == ["row_id", "probability", "label"]
    assert list(coefficients["term"]) == TERMS
    assert np.isfinite(coefficients["Estimate"]).all()


def test_model_interface_shapes():
    frame = synthetic()
    model = new_model.Model().fit(frame)
    probabilities = np.asarray(model.predict_proba(frame.iloc[:7]))
    assert probabilities.shape == (7,)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    assert set(model.get_coefficients()) == set(TERMS)
