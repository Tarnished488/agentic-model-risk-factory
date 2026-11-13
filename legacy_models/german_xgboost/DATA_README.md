# German Credit Risk input (XGBoost + SHAP)

This example contains the original R Notebook and its provenance. Download german_credit_data.csv from the dataset page under its applicable terms. Keep borrower-level data local. Discover reads CSV headers only and does not run R or train models. The Notebook refers to /kaggle/input/datasets/kabure/german-credit-data-with-risk paths, which must be adapted when later running it locally.

The notebook tunes logistic regression, random forest and XGBoost workflows with tidymodels on a stratified 80/20 split (set.seed(42)) and explains XGBoost with SHAP. It also tabulates sex and age groups, which makes it the fairness case. No trained model file or reference predictions have been supplied. Business owner, operational scope and approved model risk tier remain unconfirmed.
