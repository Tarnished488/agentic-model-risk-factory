# Give Me Some Credit input (probit GLM)

This example contains the original R Markdown script and its provenance. It uses the same competition data as gmsc_lightgbm: download cs-training.csv, cs-test.csv and Data Dictionary.xls from the competition page under its applicable terms. Keep borrower-level data local. Discover reads CSV headers only and does not run R or train models. The script refers to /kaggle/input/GiveMeSomeCredit paths, which must be adapted when later running it locally.

The script renames columns, filters records (age > 0, 90-day lateness codes 96/98, DebtRatio > 1, utilisation > 1), imputes Dependents with 0 and MonthlyIncome with the median, draws a 70/30 split with set.seed(123), and fits one probit GLM on five variables. No trained model file or reference predictions have been supplied. Business owner, operational scope and approved model risk tier remain unconfirmed.
