# Agentic Model Risk Factory

EXL × Trinity Data & AI Hackathon 2026, Challenge 04.

## Problem

Banks still run credit risk models written years ago in R or SAS. Moving them to Python is slow and risky: the documentation is thin or wrong, a translation can run cleanly yet change what the model predicts, and every step needs evidence a model validator can check.

We ask whether AI agents can document, rebuild and validate a legacy model faster, while every claim stays traceable to the source and every decision stays with a human.

## What it does

Four agents run in sequence. Each one stops at a human gate.

```text
legacy code ─► Discover ─► Gate 1 ─► Document ─► Gate 2 ─► Redevelop ─► Gate 3 ─► Validate ─► Gate 4
               inventory            6-chapter doc          IR, Python,             metrics,
               + evidence           + self-check           tests, parity           Word report
```

| Agent | Output | Human gate |
|---|---|---|
| Discover | Model inventory with source evidence and unknowns | Gate 1: check each field, correct mistakes |
| Document | Six-chapter model document; every claim cites source lines and is self-checked | Gate 2: approve, edit or reject the document |
| Redevelop | Intermediate specification (IR), Python code and tests; parity against the R reference, up to 3 retries | Gate 3: review IR, code and parity; registers the rebuild as Staging |
| Validate | Discrimination, calibration, stability, fairness and coefficient-sign checks, red/amber/green ratings, LLM explanation with number checking, Word report | Gate 4: approve for research use only |

- Gates 1–2 form **Review 1** (do we understand the model?); gates 3–4 form **Review 2** (is the rebuild right and fit for use?).
- A rejection reruns the stage. At gate 3 or 4 the reviewer can instead **return the model to Review 1** when the approved document is wrong.
- Stage-by-stage design: [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md).
- Architecture diagram: [docs/architecture.svg](docs/architecture.svg) (PNG: [docs/architecture.png](docs/architecture.png)).
- Every approval is a SHA-256 snapshot. Downstream stages refuse input that changed after approval.
- Hard rules the reviewer cannot override: no approval without passed parity (gate 3) or with a red finding (gate 4).

## Models

| Model | Source (Kaggle, Apache 2.0) | Documented | AI rebuild + validation |
|---|---|---|---|
| `gmsc_probit` Probit GLM | sunnyboyngobeni | Yes | Yes: validated for research, overall amber |
| `gmsc_lightgbm` LightGBM | brynning | Yes | No |
| `german_xgboost` XGBoost + SHAP | hafsarouchdi | Yes | No |
| `gmsc_gbm` caret GBM | fastwalker | Yes | No |

Probit result: DeepSeek generated the IR, code and tests, and parity passed after one retry. On the 33,607-row holdout, AUC is 0.8231 and KS is 0.4919. Age-group fairness metrics fall in the warning band, and sex is unavailable in the data, so the overall rating is amber.

A real finding: the German XGBoost notebook reports AUC 0.2729 because its evaluator treats "good" as the event. With the event direction corrected it is 0.7271. Review 1 missed this at first; the document was corrected, and Validate now flags any AUC below 0.5 as a possible direction error.

## Quick start

Python 3.10+.

```bash
python -m pip install -r requirements.txt
python -m pytest -q                                   # 93 tests, no API key needed
```

Run the pipeline. Without `--llm` the agents use hand-written baselines offline.

```bash
python -m src.pipeline status
python -m src.pipeline run --model gmsc_probit [--llm]           # runs until the next gate
python -m src.pipeline review --model gmsc_probit --reviewer "Discover reviewer" \
  --decision approve --notes "Checked against the source"
python -m src.pipeline review ... --decision reject --return-to-review1 --issue "..."   # gates 3-4
```

Or use the review console and dashboard:

```bash
streamlit run src/ui/app.py
```

For a public deployment (e.g. Streamlit Community Cloud, entry file `src/ui/app.py`), set the secret `SANDBOX = "1"`. The app then generates synthetic rows in the Give Me Some Credit layout, uses the Python baseline as the parity reference, runs offline and keeps its own artifact folder, so every stage runs without real data, R or an API key. Results describe random data, not the real model.

LLM mode needs an OpenAI-compatible endpoint in a local `.env` (never commit it):

```text
LLM_BASE_URL=<provider base URL>
LLM_MODEL=<model name>
LLM_API_KEY=<your key>
```

Parity needs the R reference outputs. Install R 4.4+, then:

```bash
python -m scripts.download_gmsc           # Give Me Some Credit into data/GiveMeSomeCredit/ (local only)
Rscript scripts/gmsc_probit_export.R      # writes data/reference/gmsc_probit/
```

## Repository layout

```text
src/
  pipeline.py        end-to-end runner: run / review / status / registry
  discover/          inventory extraction and gate 1
  document/          reverse-engineered documentation, self-check and gate 2
  redevelop/         IR, code and test generation, sandbox, parity and gate 3
  validate/          metrics, fairness, ratings, explainer, Word report and gate 4
  ui/                Streamlit console and dashboard
config/              pipeline settings and validation policy (thresholds)
legacy_models/       the four source notebooks with provenance
scripts/             R reference exports, data download, offline demo
docs/                Responsible AI statement and run evidence
tests/
```

## Responsible AI

See [docs/responsible_ai.md](docs/responsible_ai.md). In short: feature rows never go to the LLM (parity retries share only row IDs and scores of the worst-matching rows), numbers are computed by code and an LLM explanation cannot decide pass or fail, every claim cites its source, and approval means research use only, never production. Reviewer identity is self-declared and post-deployment monitoring is not built.

## AI tools used

- DeepSeek is the LLM inside the Discover, Document, Redevelop and Validate agents.
- AI coding assistants (Codex and Claude Code) helped with coding and debugging.

## Licences

Our code is original team work. The legacy notebooks are third-party Apache 2.0 works; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Datasets are not included; obtain them under their own terms.
