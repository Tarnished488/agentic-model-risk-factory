# Responsible AI note

Agentic Model Risk Factory, team Onlyfours, EXL × Trinity Data & AI Hackathon 2026, Challenge 04.

## Data used

- **Public data only.** Give Me Some Credit (Kaggle competition, 150,000 rows) and German Credit (1,000 rows). There is no personal or client data. The download script checks pinned SHA-256 hashes of a public mirror.
- **Legacy models.** Four public Kaggle notebooks under Apache 2.0, credited in `THIRD_PARTY_NOTICES.md`.
- **Public demo.** The deployed app runs a sandbox on synthetic rows in the Give Me Some Credit layout, with no API key.

## Privacy and security

- **Feature rows never go to the LLM.** Agents send source code, documents and CSV column headers; notebook outputs are dropped. One exception: after a failed parity round, the retry feedback includes the IDs and R/Python scores of the worst-matching rows (no feature values).
- **Borrower-level data stays local.** `data/` and per-row predictions are never committed. Reports, charts and exported evidence use aggregates only.
- **Generated code is contained.** It must pass an AST allow-list (no file, network, `os`, `subprocess` or `eval` access) and runs in an isolated subprocess with a timeout and without the API key.
- **Source is treated as data.** The Discover, Redevelop and Validate prompts tell the model to ignore any instructions found inside the legacy material. The API key lives in a local, git-ignored `.env`, and the console binds to localhost.

## Bias and fairness

- **Checks.** With fairlearn, per protected attribute: TPR gap, demographic parity difference, disparate impact (four-fifths rule) and equalized odds difference, at the legacy cut-off of 0.2. Groups need at least 100 rows and 20 per class.
- **Results.** On Probit, the age (< 40 vs ≥ 40) TPR gap is 0.106 and the demographic parity difference is 0.052. Both are inside the warning band, so the model is rated amber. Disparate impact is 0.946 (passes).
- **Gaps.** Give Me Some Credit has no sex attribute, so sex is reported as unverifiable, not as a pass. German Credit's 200-row test set is too small for any fairness conclusion. These metrics show group differences; they cannot establish legal compliance, and the cut-off is not a lending policy.

## Known limitations

- The data may not represent any bank's applicants. It has no time variable, so there is no out-of-time test; PSI only compares train and holdout.
- Only the Probit model was rebuilt by AI. The other three are documented, not rebuilt.
- The LLM makes mistakes. A human-approved document reversed the German XGBoost AUC direction (0.27 reported vs 0.73 correct). Generated code also used a wrong clamp sign and named the target as the row id. Each was caught later by parity, Validate or review, then fixed.
- Legacy defects such as leakage and preprocessing fitted before the split are reproduced on purpose for parity, and reported rather than fixed.

## Safeguards

- **Evidence.** Every extracted field and document claim cites source lines, which code checks verbatim. Unknowns stay unknown.
- **Code decides numbers.** Parity against the R reference (AUC difference ≤ 0.005, KS difference ≤ 0.01, coefficients ≤ 1e-6) and all metrics are computed by code. Every number in the LLM explanation must match the metrics.
- **Human gates with hard rules.** Four gates need a named reviewer. Gate 3 cannot approve a failed parity run, and gate 4 cannot approve a red finding. Amber findings need written resolution notes.
- **Correction routes.** Gate 3 or 4 can return a model to Review 1 when the document is wrong. The latest approval can be withdrawn with a reason.
- **Traceability.** Each decision is a SHA-256 snapshot that the next stage re-verifies. Every LLM call is logged with model, prompt version, input hash and tokens.

**Residual risk.** Human review can miss errors, as the AUC case shows. Reviewer identity is self-declared and nothing enforces independence between gates. Hashes detect changes but are not signatures. Approval means research use only. Monitoring and bank-specific policy are future work.

## AI tools declared

- **DeepSeek** is the LLM inside the four agents.
- **AI coding assistants (Codex, Claude Code)** helped write, debug and review code and documentation. Commits co-written with Claude Code carry a `Co-Authored-By` line.

The team reviewed every result and every gate decision.
