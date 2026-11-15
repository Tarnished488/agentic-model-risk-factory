# GMSC GBM source

Original notebook: https://www.kaggle.com/code/fastwalker/give-me-credit (version 13).
The notebook is downloaded locally by scripts.prepare_scoring_references; it is not redistributed while license verification remains open. Use the GMSC data prepared by scripts.download_gmsc. The original uses a 30,000-row training sample; the remaining 120,000 labelled rows form the additional migration holdout. This holdout is not the unlabeled competition test.

Compatibility updates: select the first ID column, use step_impute_mean/update_role, and run serially. Original CV index semantics and pre-CV preprocessing are preserved and require review. The original submission selected good-event probability; this migration explicitly returns bad-event probability. No Python retraining or automatic LLM conversion is claimed.
