# Agent workflow

How the four agents and four human gates process one legacy model. The last table lists where the implementation differs from the original plan.

## Overall flow

```text
Input: legacy_models/<model>/ (code, notebooks, documents) + dataset
   │
   ▼
① Discover ── inventory.json ──► Gate 1
   │
   ▼
② Document ── model document + consistency report ──► Gate 2
   │
   ▼
③ Redevelop ── IR → tests → code → sandbox → parity
   │   parity fails and retries < 3 → feed differences back, regenerate code
   │   parity fails after 3 retries → difference report, human takes over
   ▼
Gate 3 ── registry: Staging
   │
   ▼
④ Validate ── metrics + explainer + Word report ──► Gate 4 ── Validated (research)

Throughout: audit_log.jsonl for every LLM call; optional MLflow run per stage.
```

A rejection reruns the stage before the gate. At gate 3 or 4 the reviewer can instead return the model to Review 1 (gate 2) when the approved document is wrong; the rebuild then reruns on the corrected document.

## Shared state

| Field | Meaning |
|---|---|
| `model_id` | Model being processed |
| `inventory_record` | Its inventory entry |
| `doc_path` | Approved document |
| `ir_path` | Intermediate representation (language-neutral logic) |
| `generated_code_path` | Generated Python model |
| `parity_result` | Parity history, one entry per round |
| `retry_count` | Parity retries used |
| `metrics_path` / `report_path` | Validation outputs |
| `approvals` | Gate decisions |
| `status` | Current stage status |

## ① Discover: build the inventory

**Input:** one folder per model with `.R`, `.Rmd`, `.py`, `.sas`, `.ipynb`, `.md`, `.txt`, `.json`, `.csv`, `.docx` or `.pdf` files.

1. **Scan (code):** walk each model folder. Notebooks keep code and Markdown cells; CSV files contribute headers only.
2. **Extract (LLM):** name, business type, language, algorithm, inputs, derived features, target, output, training stages, packages, owner and version. Every field cites source lines.
3. **Check (code):** Pydantic validates the JSON and quotes must match the source. Up to two retries with the error, then fall back to the offline extraction marked `needs_attention`.
4. **Documentation status:** `needs_review`, `missing`, or `current`/`outdated` only with content evidence. File dates are a hint, not evidence.
5. **Risk tier (rule):** provisional, and only when the use is stated (credit decision high, reporting medium, research low).

**Output:** `inventory.json`, one record per model, with evidence and unknowns.

**Gate 1:** the reviewer checks each field against the source and types corrections. The locked record is not edited; corrections travel with the approval to Document.

## ② Document: reverse-engineer the documentation

**Input:** the approved inventory record, source code, data README and existing documents.

1. **Chunk (code):** split code into data, transformation, fitting and scoring windows within a size budget.
2. **Write by chapter (LLM):** six chapters aligned with SS1/23: purpose and scope; data and sample; variables and transformations; methodology; assumptions and limitations; use and monitoring.
3. **Cite:** every claim cites source lines, such as `[original.Rmd:L42-58]`.
4. **Self-check (second LLM pass):** each claim is checked against the source as consistent, inconsistent or unverifiable.

**Output:** `docs/<model_id>.md` and `consistency_report.json`.

**Gate 2:** the reviewer reads the draft against the source, may edit it, and must write resolution notes for edits or unresolved items. Approval locks the document as Redevelop's input.

## ③ Redevelop: rebuild in Python

**Input:** source code, the approved document, data column headers, and on retry the previous parity differences. Data rows never go to the LLM.

1. **IR (LLM):** a language-neutral specification of filters, preprocessing, model and scoring, with source evidence. Writing the IR before code is more reliable and easier to review.
2. **Tests (LLM):** pytest from the IR. When a hand-written baseline exists, the tests must pass on it first ("test the tests").
3. **Code (LLM):** a Python module with `fit / predict_proba / get_coefficients` and `run(...)`.
4. **Sandbox (code):** an AST allow-list (numpy, pandas, scipy; no os, subprocess, eval or file access), then pytest in an isolated subprocess without the API key, then a run on the real data.
5. **Parity (code, no LLM):** compare with the R reference. Inference tier: maximum absolute difference ≤ 1e-6. Re-estimation tier: AUC difference ≤ 0.005, KS difference ≤ 0.01, Spearman ≥ 0.99. GLMs also compare coefficients (≤ 1e-6).
6. **Route:** pass → gate 3; fail with retries < 3 → regenerate code with the differences; fail after 3 retries → `difference_report.md` for a human.

**Output:** `ir.json`, `new_model.py`, `tests/test_new_model.py`, `parity_result.json`.

**Gate 3:** the reviewer compares the IR and code with the legacy source and the approved document, and checks parity. Only a passed run can be approved. Approval registers the rebuild as Staging.

## ④ Validate: independent validation

Deterministic calculation first, LLM explanation second.

1. **Performance (code):** AUC, Gini, KS, Brier, log loss, calibration error and table. AUC below 0.5 is flagged as a possible reversed score direction.
2. **Stability (code):** PSI per feature between train and holdout.
3. **Fairness (code, fairlearn):** TPR gap, demographic parity difference, disparate impact (four-fifths rule) and equalized odds difference per protected attribute. Missing attributes or small groups are unverifiable, never a pass.
4. **Conceptual soundness (code):** coefficient signs against the expected business direction.
5. **Ratings (code):** red, amber or green per check against `config/validation_policy.json`, plus an overall rating.
6. **Explainer (LLM):** writes the narrative and a conclusion from the metrics only.
7. **Number check (code):** every number in the narrative must appear in the metrics, and the conclusion must agree with the rules; otherwise retry, then fall back to a template.
8. **Report (code):** Markdown, HTML and Word, with charts built from aggregates only.

**Gate 4:** red findings block approval; amber findings and limitations need resolution notes. Approval means validated for research use only.

## Cross-cutting

| Concern | Handling |
|---|---|
| LLM output malformed | Pydantic check, retry with the error, at most 2 retries |
| Generated code fails | Error lines go back to code generation |
| Parity fails | Difference feedback, at most 3 retries, then a human |
| API timeout | Exponential backoff; validated response cache |
| Tampering | SHA-256 snapshots at each gate; the next stage re-verifies them |
| Audit | One JSONL line per LLM call: stage, model, prompt version, input hash, tokens |

## As built: differences from the original plan

| Plan | Implementation |
|---|---|
| LangGraph `interrupt` for gates | A resumable state machine (`src/pipeline.py`), one state file per model, driven by the CLI or the Streamlit console |
| PDF validation report | Word, Markdown and HTML |
| Gate 4 approves for Production | Gate 4 approves for research use only |
| Out-of-time backtest and per-period metrics | Not possible: the data has no time variable, so this is listed as an unknown |
| Data dictionary as input | Data README and CSV headers; Excel dictionaries must be saved as CSV first |
| Group files by name or header comments | One folder per model, to avoid false merges |
| Monitor agent (optional) | Not built |
| Rejection returns to the previous stage | Same, plus a return from gates 3–4 to gate 2 when the document is wrong |
