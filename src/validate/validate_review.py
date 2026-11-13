"""Gate 4: explicit research validation review, never production credit approval."""
import argparse
import json
import uuid
from pathlib import Path
from src.discover.discover_agent import atomic_json, digest, file_sha256, now, output_lock
from src.redevelop.redevelop_review import load_validate_input


def verify_validation(state_path: Path):
    state = json.loads(state_path.read_text(encoding='utf-8'))
    upstream_path = Path(state['redevelop_review_path'])
    if file_sha256(upstream_path) != state['upstream_review_sha256']:
        raise ValueError('Upstream review changed')
    upstream = load_validate_input(upstream_path)
    if any(state.get(k) != v for k, v in upstream.items() if k not in ('status', 'metrics_path', 'report_path')):
        raise ValueError('Validation lineage differs from gate 3')
    report = json.loads(Path(state['metrics_path']).read_text(encoding='utf-8'))
    if (digest(report) != state['metrics_sha256'] or report['model_id'] != state['model_id']
            or report['status'] != state['status']
            or file_sha256(Path(state['report_path'])) != state['report_sha256']):
        raise ValueError('Validation metrics, report or state changed')
    return state, report


def review_validation(state_path: Path, reviewer: str, decision: str, notes: str, resolution_notes: str = '') -> Path:
    if decision not in ('approve', 'reject') or not reviewer.strip() or not notes.strip():
        raise ValueError('Reviewer, decision and review notes are required')
    with output_lock(state_path.resolve().parent):
        state, report = verify_validation(state_path)
        if decision == 'approve':
            if state['status'] != 'pending_validation_review' or any(f['result'] == 'fail' for f in report['findings']):
                raise ValueError('Failed validation rules require remediation and rerun')
            if not resolution_notes.strip():
                raise ValueError('Explicit resolution of limitations and unverified checks is required')
        directory = state_path.resolve().parent / 'reviews' / str(uuid.uuid4())
        directory.mkdir(parents=True)
        approval = {'gate': 'validate', 'reviewer': reviewer, 'decision': decision, 'notes': notes,
                    'resolution_notes': resolution_notes, 'time': now(), 'metrics_sha256': state['metrics_sha256'],
                    'scope': 'research validation only; no deployment authorization'}
        snapshot = {**state, 'status': 'validated_for_research' if decision == 'approve' else 'rejected_validation',
                    'approval': approval, 'approvals': [*state['approvals'], approval], 'report': report,
                    'report_text': Path(state['report_path']).read_text(encoding='utf-8')}
        snapshot['snapshot_sha256'] = digest(snapshot)
        path = directory / 'review.json'
        atomic_json(path, snapshot)
        return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('state', type=Path)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--decision', choices=['approve', 'reject'], required=True)
    p.add_argument('--notes', required=True)
    p.add_argument('--resolution-notes', default='')
    a = p.parse_args()
    try:
        print(review_validation(a.state, a.reviewer, a.decision, a.notes, a.resolution_notes))
        return 0
    except (ValueError, KeyError, OSError) as e:
        print(f'Validation review stopped: {e}')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
