"""Deterministic validation. No LLM, training or credit decisions. Gate 3 is mandatory."""
from __future__ import annotations
import argparse
import html
import json
import math
import uuid
from pathlib import Path

import os

import numpy as np
import pandas as pd
from src.discover.discover_agent import atomic_json, audit, call_llm, digest, file_sha256, now
from src.redevelop.parity import auc, ks
from src.redevelop.redevelop_review import load_validate_input
from src.validate.charts import draw_charts, log_validation_mlflow
from src.validate.explainer import llm_explanation, template_explanation
from src.validate.fairness import fairness_audit
from src.validate.ratings import annotate
from src.validate.report_docx import write_report

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / 'config/validation_policy.json'


def read_policy(path: Path) -> dict:
    p = json.loads(path.read_text(encoding='utf-8'))
    for k in ('min_auc', 'min_ks', 'max_ece', 'max_tpr_gap', 'decision_threshold'):
        if not isinstance(p[k], (float, int)) or isinstance(p[k], bool) or not 0 <= p[k] <= 1:
            raise ValueError(f'Invalid policy field: {k}')
    for k in ('max_feature_psi', 'min_group_size', 'min_group_class_size'):
        if not isinstance(p[k], (float, int)) or not math.isfinite(p[k]) or p[k] <= 0:
            raise ValueError(f'Invalid policy field: {k}')
    if not math.isfinite(p['fairness_split']):
        raise ValueError('fairness_split must be finite')
    p.setdefault('max_demographic_parity_difference', 0.1)
    p.setdefault('min_disparate_impact', 0.8)
    p.setdefault('max_equalized_odds_difference', 0.2)
    for spec in p.get('fairness_attributes', []):
        if not spec.get('column') or ('split' in spec and not math.isfinite(spec['split'])):
            raise ValueError('Each fairness attribute needs a column and an optional finite split')
    if not all(sign in ('+', '-') for sign in p.get('expected_signs', {}).values()):
        raise ValueError('expected_signs values must be "+" or "-"')
    return p


def coefficient_signs(state: dict, ir: dict, policy: dict) -> tuple[dict, list[dict]]:
    """Conceptual soundness (design doc 4.4): code decides each sign; the explainer only describes it."""
    path = state.get('coefficients_path')
    expected = policy.get('expected_signs', {})
    if not path or not Path(path).is_file():
        return {}, [{'check': 'coefficient_signs', 'result': 'unverifiable', 'severity': 'medium',
                     'reason': 'No coefficients in the handoff (e.g. tree models); review drivers via SHAP instead'}]
    raw_by_alias = {alias: raw for raw, alias in ir.get('rename', {}).get('mapping', {}).items()}
    coefficients = {}
    for row in pd.read_csv(path).itertuples(index=False):
        name = str(row.term).strip('`')
        if name != '(Intercept)':
            coefficients[raw_by_alias.get(name, name)] = float(row.Estimate)
    findings = []
    for feature in ir['model']['features']:
        if feature not in expected:
            continue
        if feature not in coefficients:
            findings.append({'check': 'coefficient_sign:' + feature, 'result': 'unverifiable', 'severity': 'medium',
                             'reason': 'Feature has no coefficient in the rebuilt model'})
            continue
        value = coefficients[feature]
        passed = value > 0 if expected[feature] == '+' else value < 0
        findings.append({'check': 'coefficient_sign:' + feature, 'result': 'pass' if passed else 'fail',
                         'severity': 'medium', 'value': value, 'limit': expected[feature]})
    return coefficients, findings


def read_ids(path: Path, column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={column: str})
    if column not in frame or frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f'Missing or duplicate {column} in {path.name}')
    return frame


def performance(y, scores) -> dict:
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    if len(y) == 0 or not set(y) <= {0, 1} or len(set(y)) != 2:
        raise ValueError('Evaluation needs non-empty binary labels with both classes')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid probabilities')
    bins = np.minimum((scores * 10).astype(int), 9)
    calibration = []
    for i in range(10):
        mask = bins == i
        if mask.any():
            calibration.append({'bin': i, 'count': int(mask.sum()),
                                'mean_probability': float(scores[mask].mean()),
                                'observed_rate': float(y[mask].mean())})
    ece = sum(b['count'] * abs(b['mean_probability'] - b['observed_rate']) for b in calibration) / len(y)
    # Clipping only for log-loss numerical evaluation; model predictions are unchanged.
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    return {'rows': len(y), 'events': int(y.sum()), 'event_rate': float(y.mean()),
            'auc': auc(scores.tolist(), y.tolist()), 'gini': 2 * auc(scores.tolist(), y.tolist()) - 1,
            'ks': ks(scores.tolist(), y.tolist()), 'brier': float(np.mean((scores - y) ** 2)),
            'constant_rate_brier': float(y.mean() * (1 - y.mean())),
            'log_loss': float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            'ece': float(ece), 'calibration': calibration}


def psi(reference: pd.Series, current: pd.Series) -> float | None:
    """Numeric train/holdout distribution comparison, with a separate missing bin."""
    a, b = pd.to_numeric(reference, errors='coerce'), pd.to_numeric(current, errors='coerce')
    a, b = a.replace([np.inf, -np.inf], np.nan), b.replace([np.inf, -np.inf], np.nan)
    finite = a.dropna()
    if len(a) == 0 or len(b) == 0 or finite.empty:
        return None
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))
    inner = edges[1:-1] if len(edges) > 1 else edges
    edges = np.r_[-np.inf, inner, np.inf]
    ca = np.r_[np.histogram(a.dropna(), edges)[0], a.isna().sum()].astype(float)
    cb = np.r_[np.histogram(b.dropna(), edges)[0], b.isna().sum()].astype(float)
    pa, pb = (ca + 0.5) / (ca.sum() + 0.5 * len(ca)), (cb + 0.5) / (cb.sum() + 0.5 * len(cb))
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def evaluate(state: dict, policy: dict) -> dict:
    ir = json.loads(Path(state['ir_path']).read_text(encoding='utf-8'))
    raw = pd.read_csv(state['data_path'], dtype=str)
    row_col = ir['data']['row_id']
    # R/readr/pandas give the empty first CSV header different names.
    if row_col not in raw and row_col in ('row_id', 'Applicant', 'X', '...1', 'Unnamed: 0', '',
                                         'first unnamed column (readr names it ...1, renamed to Applicant)'):
        row_col = raw.columns[0]
    if row_col not in raw:
        raise ValueError('IR row identifier is absent from raw data')
    if raw[row_col].isna().any() or raw[row_col].duplicated().any():
        raise ValueError('Raw identifiers must be non-null and unique')
    raw = raw.set_index(row_col)
    target = ir['model']['target']
    if target not in raw:
        raise ValueError('IR target absent from raw data')
    train = read_ids(Path(state['train_ids_path']), 'row_id')
    predictions = read_ids(Path(state['holdout_predictions_path']), 'row_id')
    required = {'probability', 'label'}
    if not required <= set(predictions):
        raise ValueError('Holdout predictions require row_id, probability, label')
    train_ids, holdout_ids = set(train.row_id), set(predictions.row_id)
    if train_ids & holdout_ids:
        raise ValueError('Training/holdout leakage: overlapping row identifiers')
    if not train_ids or not holdout_ids or not (train_ids | holdout_ids) <= set(raw.index):
        raise ValueError('Split is empty or contains identifiers absent from raw data')
    eligible = pd.Series(True, index=raw.index)
    operations = {'<': lambda x, y: x < y, '<=': lambda x, y: x <= y,
                  '>': lambda x, y: x > y, '>=': lambda x, y: x >= y,
                  '==': lambda x, y: x == y, '!=': lambda x, y: x != y}
    for rule in ir.get('filters', []):
        values = pd.to_numeric(raw[rule['column']], errors='coerce')
        if rule['compare_as'] == 'string':
            values = values.map(lambda x: f'{x:.15g}' if pd.notna(x) else None)
        eligible &= operations[rule['op']](values, rule['value']).fillna(False)
    expected = set(raw.index[eligible])
    if train_ids | holdout_ids != expected:
        raise ValueError('Split does not cover exactly the IR-eligible population')
    yraw = pd.to_numeric(raw.loc[predictions.row_id, target], errors='raise').to_numpy(float)
    yreported = pd.to_numeric(predictions.label, errors='raise').to_numpy(float)
    if not np.isfinite(yraw).all() or not np.array_equal(yraw, yreported) or not set(yraw) <= {0, 1}:
        raise ValueError('Prediction labels do not match raw binary labels')
    scores = pd.to_numeric(predictions.probability, errors='raise').to_numpy(float)
    metrics = performance(yraw, scores)
    findings = []
    def check(name, passed, value, limit, severity='high'):
        findings.append({'check': name, 'result': 'pass' if passed else 'fail',
                         'severity': severity, 'value': value, 'limit': limit})
    check('holdout_auc', metrics['auc'] >= policy['min_auc'], metrics['auc'], policy['min_auc'])
    check('holdout_ks', metrics['ks'] >= policy['min_ks'], metrics['ks'], policy['min_ks'])
    check('calibration_ece', metrics['ece'] <= policy['max_ece'], metrics['ece'], policy['max_ece'], 'medium')
    distribution = {}
    for feature in ir['model']['features']:
        if feature not in raw:
            raise ValueError(f'Model feature absent from raw data: {feature}')
        value = psi(raw.loc[train.row_id, feature], raw.loc[predictions.row_id, feature])
        distribution[feature] = value
        if value is not None:
            check('train_holdout_psi:' + feature, value <= policy['max_feature_psi'], value, policy['max_feature_psi'], 'medium')
    groups, fairness, fairness_findings, fairness_unknowns = fairness_audit(
        raw, predictions.row_id, yraw, scores, policy)
    findings += fairness_findings
    coefficients, sign_findings = coefficient_signs(state, ir, policy)
    findings += sign_findings
    unknowns = ['Business scope, representativeness and policy thresholds require human confirmation.',
                'Train/holdout PSI is sample comparison, not temporal drift monitoring.',
                'Group rates are exploratory; decision threshold is illustrative, not a lending policy.',
                'Coverage matches IR filters; appropriateness of those exclusions requires review.',
                'No production data, external validation or regulatory compliance certification.',
                'No time variable in the data, so out-of-time back-testing and per-period metrics are not possible.']
    unknowns += fairness_unknowns
    if any('before the split' in (i.get('computed_on') or '').lower() for i in ir.get('imputation', [])):
        unknowns.append('Legacy preprocessing may fit imputation before the split; independently review leakage.')
    overall_rating = annotate(findings, policy)
    return {'metrics': metrics, 'train_holdout_psi': distribution, 'groups': groups, 'fairness': fairness,
            'findings': findings, 'overall_rating': overall_rating, 'coefficients': coefficients,
            'curves': curves(yraw, scores), 'unknowns': unknowns, 'train_rows': len(train), 'raw_rows': len(raw),
            'status': 'needs_attention' if any(f['result'] == 'fail' for f in findings) else 'pending_validation_review'}


def curves(y: np.ndarray, scores: np.ndarray, points: int = 101) -> dict:
    """Aggregate ROC and score histograms for charts; no row-level values leave this function."""
    order = np.argsort(-scores, kind='stable')
    tp, fp = np.cumsum(y[order]), np.cumsum(1 - y[order])
    picks = np.unique(np.linspace(0, len(y) - 1, points).astype(int))
    roc = [{'fpr': 0.0, 'tpr': 0.0}] + [{'fpr': float(fp[i] / fp[-1]), 'tpr': float(tp[i] / tp[-1])} for i in picks]
    edges = np.linspace(0, 1, 21)
    return {'roc': roc, 'histogram_edges': edges.tolist(),
            'histogram_good': np.histogram(scores[y == 0], edges)[0].tolist(),
            'histogram_bad': np.histogram(scores[y == 1], edges)[0].tolist()}


def render_report(report: dict) -> tuple[str, str]:
    lines = ['# Independent model validation', '', f"Model: {report['model_id']}",
             f"Mode: {report['mode']} | Status: {report['status']}",
             f"Policy: {report['policy']['version']} (illustrative research rules)", '',
             'Parity checks translation fidelity. This report separately evaluates holdout performance.', '',
             '## Metrics', '', '| Metric | Value |', '| --- | --- |']
    for key, value in report['metrics'].items():
        if key != 'calibration':
            lines.append(f'| {key} | {value} |')
    lines += ['', '## Findings', '']
    lines += [f"- {f['check']}: {f['result']} ({f['severity']})" for f in report['findings']]
    lines += ['', '## Limitations and review actions', ''] + ['- ' + s for s in report['unknowns']]
    explanation = report.get('explanation')
    if explanation:
        lines += ['', f"## Explanation ({report['explainer_mode']}; numbers checked against metrics)", '',
                  f"Conclusion: {explanation['conclusion']} - {explanation['conclusion_reason']}", '']
        for section in ('summary', 'performance', 'calibration', 'stability', 'fairness', 'conceptual_soundness'):
            lines += [f"### {section.replace('_', ' ').title()}", '', explanation[section], '']
    md = '\n'.join(lines) + '\n'
    rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v:.6g}</td></tr>' for k, v in report['metrics'].items() if isinstance(v, (int, float)))
    findings = ''.join(f"<li><b>{html.escape(f['check'])}</b> · {f['result']}</li>" for f in report['findings'])
    calibration = ''.join(f"<tr><td>{b['bin']}</td><td>{b['count']}</td><td>{b['mean_probability']:.4f}</td><td>{b['observed_rate']:.4f}</td></tr>" for b in report['metrics']['calibration'])
    groups = ''.join('<li>' + html.escape(json.dumps(g)) + '</li>' for g in report['groups'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model validation report</title><style>body{{font:16px system-ui;margin:40px auto;max-width:1050px;padding:0 24px;color:#172b4d;background:#f4f7fb}}section{{background:white;padding:24px;margin:18px 0;border-radius:14px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dde3ed}}h1{{font-size:34px}}.tag{{color:#675300;background:#fff1bf;padding:10px;border-radius:8px}}nav a{{margin-right:20px}}</style>
<h1>Independent model validation</h1><p>{html.escape(report['model_id'])} · {html.escape(report['mode'])}</p><p class="tag">{report['status']} · Research evidence, pending human judgement</p>
<nav><a href="#metrics">Metrics</a><a href="#findings">Findings</a><a href="#calibration">Calibration</a><a href="#limits">Review actions</a></nav>
<section id="metrics"><h2>Holdout performance</h2><table>{rows}</table></section>
<section id="findings"><h2>Rule checks</h2><ul>{findings}</ul></section>
<section id="calibration"><h2>Calibration</h2><table><tr><th>Bin</th><th>Rows</th><th>Predicted</th><th>Observed</th></tr>{calibration}</table></section>
<section><h2>Exploratory groups</h2><ul>{groups}</ul><h2>Train / holdout PSI</h2><pre>{html.escape(json.dumps(report['train_holdout_psi'],indent=2))}</pre></section>
<section id="limits"><h2>Review actions</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['unknowns'])}</ul><p>Policy: {html.escape(report['policy']['version'])}. No regulatory certification or deployment approval.</p></section></html>'''
    return md, page


def explain(report: dict, run_dir: Path, use_llm: bool, caller) -> tuple[dict, str, str | None]:
    """LLM explanation when enabled; the deterministic template otherwise or if the LLM cannot pass the checks."""
    if not use_llm:
        return template_explanation(report).model_dump(), 'template', None
    base = {'node': 'validate', 'agent': 'explainer', 'model_id': report['model_id'], 'run_id': run_dir.name}

    def log(attempt, outcome, details):
        audit(run_dir, {**base, 'event': 'llm_attempt', 'attempt': attempt, 'outcome': outcome,
                        'provider_model': os.environ.get('LLM_MODEL'), **details})
    try:
        return llm_explanation(report, caller or call_llm, log).model_dump(), 'llm', None
    except ValueError as error:
        # Never block validation on narrative text; the reviewer sees the fallback and why.
        return template_explanation(report).model_dump(), 'template_after_llm_failure', str(error)[:500]


def run_validation(review_file: Path, output: Path, policy_path: Path = DEFAULT_POLICY,
                   use_llm: bool = False, caller=None, use_mlflow: bool = False) -> Path:
    upstream = load_validate_input(review_file)
    policy = read_policy(policy_path)
    run_dir = output.resolve() / 'runs' / str(uuid.uuid4())
    run_dir.mkdir(parents=True)
    report = {'schema_version': '2', 'model_id': upstream['model_id'], 'generated_at': now(),
              'mode': 'deterministic_metrics', 'policy': policy, 'upstream_review_sha256': file_sha256(review_file),
              **evaluate(upstream, policy)}
    # Detect upstream mutations while computing as well as before computing.
    if load_validate_input(review_file) != upstream:
        raise ValueError('Upstream changed during validation')
    explanation, explainer_mode, explainer_error = explain(report, run_dir, use_llm, caller)
    report.update(explanation=explanation, explainer_mode=explainer_mode, explainer_error=explainer_error)
    atomic_json(run_dir / 'metrics.json', report)
    findings_doc = {'model_id': report['model_id'], 'generated_at': report['generated_at'],
                    'status': report['status'], 'overall_rating': report['overall_rating'],
                    'findings': report['findings'], 'unknowns': report['unknowns']}
    atomic_json(run_dir / 'findings.json', findings_doc)
    charts = draw_charts(report, run_dir / 'charts')
    md, page = render_report(report)
    (run_dir / 'report.md').write_text(md, encoding='utf-8')
    (run_dir / 'report.html').write_text(page, encoding='utf-8')
    write_report(report, explanation, upstream['approvals'], run_dir / 'report.docx', charts)
    if use_mlflow:
        try:
            log_validation_mlflow(run_dir, report, [run_dir / 'metrics.json', run_dir / 'findings.json',
                                                    run_dir / 'report.docx', *map(Path, charts.values())])
        except Exception as error:  # tracking is optional; the local files are the record
            audit(run_dir, {'node': 'validate', 'event': 'mlflow_failed', 'model_id': report['model_id'],
                            'error_type': type(error).__name__})
    state = {**upstream, 'status': report['status'], 'validation_run_id': run_dir.name,
             'redevelop_review_path': str(review_file.resolve()),
             'upstream_review_sha256': report['upstream_review_sha256'],
             'metrics_path': str(run_dir / 'metrics.json'), 'metrics_sha256': digest(report),
             'report_path': str(run_dir / 'report.md'), 'report_sha256': file_sha256(run_dir / 'report.md'),
             'html_path': str(run_dir / 'report.html'),
             'docx_path': str(run_dir / 'report.docx'), 'docx_sha256': file_sha256(run_dir / 'report.docx'),
             'explainer_mode': explainer_mode, 'explanation_conclusion': explanation['conclusion'],
             'overall_rating': report['overall_rating'], 'findings_path': str(run_dir / 'findings.json'),
             'chart_paths': charts}
    atomic_json(run_dir / 'state.json', state)
    return run_dir / 'state.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('review_file', type=Path)
    p.add_argument('--output', type=Path, default=Path('artifacts/validate'))
    p.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    p.add_argument('--llm', action='store_true', help='LLM explainer (needs LLM_BASE_URL, LLM_MODEL, LLM_API_KEY)')
    p.add_argument('--mlflow', action='store_true', help='Log metrics and artifacts to MLflow (mlflow-skinny)')
    a = p.parse_args()
    try:
        path = run_validation(a.review_file, a.output, a.policy, use_llm=a.llm, use_mlflow=a.mlflow)
        print(path)
        return 0 if json.loads(path.read_text())['status'] == 'pending_validation_review' else 2
    except (ValueError, KeyError, OSError) as e:
        print(f'Validation stopped: {e}')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
