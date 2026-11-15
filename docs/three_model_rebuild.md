# Three additional Python scoring migrations

This extends the existing pipeline through Redevelop, parity and Validate for fastwalker GBM, brynning LightGBM and German XGBoost. These are **inference migrations**: Python performs preprocessing and scoring from raw records, using portable parameters learned by the original R workflows. They do not retrain in Python and are not LLM-generated code. Probit remains the separate re-estimation example. The gate console displays this distinction.

## Reproduce

Install requirements-trees.txt in a Python 3.12 environment. R 4.4+ needs the original packages: tidyverse, recipes, caret, gbm, foreach, lightgbm, tidymodels, themis, ranger, xgboost, glmnet and jsonlite. No notebook package-install cells are executed. Existing local R packages can be selected through R_LIBS_USER.

Download GMSC with `python -m scripts.download_gmsc`. German Credit's public source is https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk ; save german_credit_data.csv under data/GermanCredit. Borrower-level files stay local.

```bash
python -m scripts.prepare_scoring_references --model gmsc_fastwalker --data data/GiveMeSomeCredit --reference data/reference/gmsc_fastwalker
python -m scripts.prepare_scoring_references --model gmsc_lightgbm --data data/GiveMeSomeCredit --reference data/reference/gmsc_lightgbm_labelled
python -m scripts.prepare_scoring_references --model german_xgboost --data data/GermanCredit --reference data/reference/german_xgboost
```

The exporter runs the relevant original training cells with fixed compatibility adaptations and writes portable model/preprocessing files, original R probabilities, fixed split IDs, a source/data/model manifest and sessionInfo. Native LightGBM/XGBoost formats contain learned parameters, not executable RDS/pickle. GBM is exported as explicit numeric trees. The current GBM API download is pinned to notebook version 13; its source is fetched locally, not redistributed while license verification is open.

On macOS, the native engines need libomp. On this machine the installed R supplies it; set `DYLD_LIBRARY_PATH=/Library/Frameworks/R.framework/Resources/lib` for the Python process if necessary. Other machines should use their normal supported native runtime installation. This does not alter global settings.

For normal human-gated execution, run `python -m src.pipeline run --model MODEL_GROUP` or use the Streamlit gate console. New settings in config/pipeline.json enable all three models. Leave `--llm` off: these reviewed adapters have not been verified with LLM code generation.

For a labelled technical rehearsal only:

```bash
python -m scripts.run_tree_rebuilds --technical-rehearsal
python -m http.server 8766 --bind 127.0.0.1 --directory outputs/three_rebuild
```

The rehearsal explicitly records Codex technical checks at gates 1–3. It never signs as a team member or approves gate 4. Teams must independently review the materials and limitations. Without --technical-rehearsal, the script pauses at the human gate.

## What is compared

- GBM: original 30,000 training rows; 120,000 previously unused labelled rows for the added holdout. Python recreates mean imputation, centering and scaling, then traverses the exported R trees. Leaf weights already include shrinkage. Tree thresholds and preprocessing parameters use 17-digit strings to preserve exact double values: shorter decimal serialization changed branches for 15 rows in the first prototype; this was fixed, not hidden by loosening tolerance.
- LightGBM: a fixed R 70/30 labelled split (105,000/45,000) is added because the original competition test has no labels. Both R fitting stages and top-100 selection remain. Python recreates raw-to-engineered features and selected feature order, then scores the R-trained native model. No copied R feature matrix is used as candidate input.
- German XGBoost: the original stratified 800/200 split, tuning grids, SMOTE and fitted recipe are retained. Python reconstructs engineered features, training factor contrasts, normalization and feature selection. Native model output is explicitly mapped to bad-event probability. Validate maps original Risk values good/bad to 0/1 and uses raw Age/Sex fields.

All comparisons use the existing inference threshold of maximum probability difference <= 1e-6. Reference parameters, data, splits and predictions are checked by manifest hashes before running; consumed assets are checked again afterwards and at the gate-3/Validate handoff. Python cannot silently substitute a different runtime model after approval.

## Validation and unresolved limitations

Validate now also checks categorical distribution shifts (smoothed category PSI/CSI) rather than silently skipping text features. Native target mapping is validated against original raw labels. Missing or unseen target values fail. Synthetic and real-reference tests cover these behaviors.

Preserved limitations are visible in reports: GBM's original CV index semantics and pre-CV preprocessing; LightGBM batch-dependent feature means and asymmetric age handling; German's pre-split high-loan quantile and insufficient fairness group/class support. No linear coefficient signs are invented for tree models. Further independent driver/SHAP review remains appropriate. Current research thresholds are illustrative, not bank policy.

All three real runs passed parity but are **amber, pending final review**. See [aggregate evidence](evidence/three_model_rebuild.json). A yellow/amber report is not deployment approval. Performance figures use different populations and splits and must not be treated as a ranking. No Python re-training, generic automatic conversion or production drift monitoring is claimed.
