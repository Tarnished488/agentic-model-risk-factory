"""Deterministic validation. No LLM, training or credit decisions. Gate 3 is mandatory."""
from __future__ import annotations
import argparse
import html
import json
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from src.discover.discover_agent import atomic_json, digest, file_sha256, now
from src.redevelop.parity import auc, ks
from src.redevelop.redevelop_review import load_validate_input

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
    return p


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
    groups = []
    col = policy['fairness_column']
    if col in raw:
        values = pd.to_numeric(raw.loc[predictions.row_id, col], errors='coerce').to_numpy()
        split = policy['fairness_split']
        for label, mask in [(f'{col} < {split}', values < split), (f'{col} >= {split}', values >= split)]:
            y, s = yraw[mask], scores[mask]
            positives, negatives = int(y.sum()), int(len(y) - y.sum())
            sufficient = len(y) >= policy['min_group_size'] and min(positives, negatives) >= policy['min_group_class_size']
            selected = s >= policy['decision_threshold']
            groups.append({'group': label, 'rows': int(len(y)), 'events': positives, 'sufficient': sufficient,
                           'selection_rate': float(selected.mean()) if len(y) else None,
                           'tpr': float(selected[y == 1].mean()) if positives else None,
                           'fpr': float(selected[y == 0].mean()) if negatives else None})
        if all(g['sufficient'] for g in groups):
            gap = abs(groups[0]['tpr'] - groups[1]['tpr'])
            check('age_group_tpr_gap', gap <= policy['max_tpr_gap'], gap, policy['max_tpr_gap'], 'medium')
        else:
            findings.append({'check': 'age_group_tpr_gap', 'result': 'unverifiable', 'severity': 'medium',
                             'reason': 'Insufficient group/class support; no fairness pass is claimed'})
    else:
        findings.append({'check': 'group_fairness', 'result': 'unverifiable', 'severity': 'medium',
                         'reason': 'Configured group column is absent'})
    unknowns = ['Business scope, representativeness and policy thresholds require human confirmation.',
                'Train/holdout PSI is sample comparison, not temporal drift monitoring.',
                'Group rates are exploratory; decision threshold is illustrative, not a lending policy.',
                'Coverage matches IR filters; appropriateness of those exclusions requires review.',
                'No production data, external validation or regulatory compliance certification.']
    if col in raw and pd.to_numeric(raw.loc[predictions.row_id, col], errors='coerce').isna().any():
        unknowns.append('Missing group values are excluded from group comparisons; review their coverage.')
    if any('before the split' in (i.get('computed_on') or '').lower() for i in ir.get('imputation', [])):
        unknowns.append('Legacy preprocessing may fit imputation before the split; independently review leakage.')
    return {'metrics': metrics, 'train_holdout_psi': distribution, 'groups': groups, 'findings': findings,
            'unknowns': unknowns, 'train_rows': len(train), 'raw_rows': len(raw),
            'status': 'needs_attention' if any(f['result'] == 'fail' for f in findings) else 'pending_validation_review'}


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


def run_validation(review_file: Path, output: Path, policy_path: Path = DEFAULT_POLICY) -> Path:
    upstream = load_validate_input(review_file)
    policy = read_policy(policy_path)
    run_dir = output.resolve() / 'runs' / str(uuid.uuid4())
    run_dir.mkdir(parents=True)
    report = {'schema_version': '1', 'model_id': upstream['model_id'], 'generated_at': now(),
              'mode': 'deterministic_no_api', 'policy': policy, 'upstream_review_sha256': file_sha256(review_file),
              **evaluate(upstream, policy)}
    # Detect upstream mutations while computing as well as before computing.
    if load_validate_input(review_file) != upstream:
        raise ValueError('Upstream changed during validation')
    atomic_json(run_dir / 'metrics.json', report)
    md, page = render_report(report)
    (run_dir / 'report.md').write_text(md, encoding='utf-8')
    (run_dir / 'report.html').write_text(page, encoding='utf-8')
    state = {**upstream, 'status': report['status'], 'validation_run_id': run_dir.name,
             'redevelop_review_path': str(review_file.resolve()),
             'upstream_review_sha256': report['upstream_review_sha256'],
             'metrics_path': str(run_dir / 'metrics.json'), 'metrics_sha256': digest(report),
             'report_path': str(run_dir / 'report.md'), 'report_sha256': file_sha256(run_dir / 'report.md'),
             'html_path': str(run_dir / 'report.html')}
    atomic_json(run_dir / 'state.json', state)
    return run_dir / 'state.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('review_file', type=Path)
    p.add_argument('--output', type=Path, default=Path('artifacts/validate'))
    p.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    a = p.parse_args()
    try:
        path = run_validation(a.review_file, a.output, a.policy)
        print(path)
        return 0 if json.loads(path.read_text())['status'] == 'pending_validation_review' else 2
    except (ValueError, KeyError, OSError) as e:
        print(f'Validation stopped: {e}')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
