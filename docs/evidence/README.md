# Offline run evidence (2026-11-13)

Real Give Me Some Credit data (third-party mirror), the original R script run independently, and the hand-written Python Probit baseline. No LLM was called, so this run does not show automatic conversion; the DeepSeek rebuild is a separate run.

- Rows: 150,000 raw, 112,021 after filters, 78,414 train, 33,607 holdout.
- Maximum R / Python difference: predictions 3.45e-14, coefficients 4.51e-14.
- AUC 0.8231, KS 0.4919, Brier 0.0464, ECE 0.0054.
- Parity passed. Gates 1–3 in this run were technical checks by an AI assistant, not team sign-offs; gate 4 was left pending.

R 4.6.1 and Python 3.12. `data_provenance.json` holds the download hashes; `summary.json` holds the aggregate metrics and artifact hashes. Borrower-level data, predictions and logs stay in the ignored `data/` and `artifacts/` folders.

This shows the hand-written rebuild reproduces R on this data and that the metrics run end to end. It does not show cross-model generalisation, out-of-time performance, full fairness or regulatory compliance.

Reproduce with `python -m scripts.download_gmsc`, `Rscript scripts/gmsc_probit_export.R` and `python -m scripts.run_offline_demo`.
