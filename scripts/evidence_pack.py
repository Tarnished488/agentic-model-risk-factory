"""Publish aggregate evidence only; raw data, predictions, paths and identities stay local."""
import argparse
import html
import json
import subprocess
from pathlib import Path
from src.discover.discover_agent import atomic_json, file_sha256, now
from src.validate.validate_review import verify_validation


def collect(state_path: Path, output: Path) -> Path:
    state, report = verify_validation(state_path)
    redevelop_path = Path(state['redevelop_review_path'])
    redevelopment = json.loads(redevelop_path.read_text(encoding='utf-8'))
    document_path = Path(redevelopment['document_review_path'])
    document = json.loads(document_path.read_text(encoding='utf-8'))
    discovery_path = Path(document['discover_review_path'])
    discovery = json.loads(discovery_path.read_text(encoding='utf-8'))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable; see build manifest'
    stages = [
        {'name': 'Discover', 'mode': discovery['inventory_record']['extraction_mode'], 'status': 'approved', 'sha256': file_sha256(discovery_path)},
        {'name': 'Document', 'mode': document['consistency_report']['mode'], 'status': 'approved', 'sha256': file_sha256(document_path)},
        {'name': 'Redevelop', 'mode': redevelopment['mode'], 'status': 'approved', 'sha256': file_sha256(redevelop_path)},
        {'name': 'Validate', 'mode': report['mode'], 'status': state['status'], 'sha256': file_sha256(state_path)}]
    artifacts = {k: {'sha256': file_sha256(Path(state[k]))} for k in
                 ['ir_path', 'generated_code_path', 'tests_path', 'data_path', 'train_ids_path', 'holdout_predictions_path', 'metrics_path', 'report_path']}
    summary = {'schema_version': '1', 'generated_at': now(), 'model_id': state['model_id'],
               'code_revision': revision, 'stages': stages, 'artifacts': artifacts,
               'mode': 'NO API — hand-written baseline rebuild; no LLM-generated code claim',
               'parity': redevelopment['parity_result'], 'validation': report,
               'gate4': 'pending; not automatically approved',
               'review_provenance': 'Gates 1–3 are Codex technical checks in this demo, not team human signatures. All human sign-off remains pending.',
               'data_origin': 'Public third-party GMSC mirror; original Kaggle bytes not independently authenticated'}
    summary['source_files_sha256'] = {str(p): file_sha256(p) for directory in ('src', 'scripts', 'config')
                                      for p in sorted(Path(directory).rglob('*'))
                                      if p.is_file() and p.suffix in ('.py', '.R', '.json')}
    # Per-row worst-case prediction examples and coefficient values are local-only.
    summary['parity'] = {k: v for k, v in summary['parity'].items() if k not in ['worst_rows', 'error']}
    if 'metrics' in summary['parity']:
        summary['parity']['metrics'] = {k: v for k, v in summary['parity']['metrics'].items() if k != 'coefficient_diffs'}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'summary.json', summary)
    cards = ''.join(f"<article><b>{html.escape(s['name'])}</b><p>{html.escape(s['mode'])}</p><span>{html.escape(s['status'])}</span></article>" for s in stages)
    m = report['metrics']
    dashboard = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model risk factory · run evidence</title>
<style>body{{font:16px system-ui;background:#f3f6fb;color:#173046;margin:40px auto;max-width:1100px;padding:0 28px}}h1{{font-size:36px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:15px}}article,section{{background:white;border:1px solid #dce4ed;padding:24px;border-radius:14px;margin-bottom:18px}}article b{{font-size:22px}}.note{{background:#fff0c4;padding:18px;border-radius:12px}}a{{color:#1457bb}}pre{{white-space:pre-wrap;word-break:break-word}}button{{padding:12px;cursor:pointer}}.metric{{font-size:30px;font-weight:700}}</style>
<h1>Model risk factory · local real-data run</h1><p>{html.escape(state['model_id'])} · Give Me Some Credit / Probit</p>
<p class="note">No API: the hand-written Python baseline is compared with real R output. The document is an offline draft; no LLM generation ran. The data is a third-party mirror, not byte-verified against the official source. Gates 1–3 are technical checks by an AI assistant, not team sign-offs; team review and gate 4 are still pending.</p>
<div class="grid">{cards}</div><div class="grid"><article>Holdout rows<p class="metric">{m['rows']:,}</p></article><article>AUC<p class="metric">{m['auc']:.4f}</p></article><article>KS<p class="metric">{m['ks']:.4f}</p></article><article>Calibration ECE<p class="metric">{m['ece']:.4f}</p></article></div>
<section><h2>What this run shows</h2><p>The hand-written Python rebuild was compared with the real R reference by parity, and independent validation recomputed the model metrics. Every file has a hash and the approval chain can be traced.</p><p>It does not show general automatic conversion by agents, that every model passes, or bank approval for production.</p><a href="validation.html">Full validation report →</a> · <a href="summary.json">Aggregate evidence JSON</a></section>
<section><h2>R / Python parity</h2><pre>{html.escape(json.dumps(summary['parity'],ensure_ascii=False,indent=2))}</pre></section>
<section><h2>Open items</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['unknowns'])}</ul></section>
<details><summary>Stage source hashes</summary><pre>{html.escape(json.dumps(stages,ensure_ascii=False,indent=2))}</pre></details></html>'''
    (output / 'index.html').write_text(dashboard, encoding='utf-8')
    (output / 'validation.html').write_text(Path(state['html_path']).read_text(encoding='utf-8'), encoding='utf-8')
    return output / 'index.html'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('state', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/demo'))
    a = p.parse_args()
    print(collect(a.state, a.output))

if __name__ == '__main__':
    main()
