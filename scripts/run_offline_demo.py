"""Reproducible real-data, no-API demonstration. Approvals are AI technical checks, not human signatures."""
import argparse
import json
from pathlib import Path
from src.discover.discover_agent import atomic_json, run_discovery
from src.discover.discover_review import review as discover_review
from src.document.document_agent import run_document
from src.document.document_review import load_redevelop_input, review_document
from src.redevelop.redevelop_agent import run_redevelop
from src.redevelop.redevelop_review import review_redevelop
from src.validate.validate_agent import run_validation
from scripts.evidence_pack import collect


def run(data: Path, reference: Path, output: Path):
    for path in [data/'cs-training.csv',reference/'train_ids.csv',reference/'holdout_predictions.csv',reference/'coefficients.csv']:
        if not path.is_file():raise ValueError(f'Required local input missing: {path}')
    output.mkdir(parents=True,exist_ok=True)
    inv=run_discovery(Path('legacy_models'),output/'discover')
    record=next(m for m in inv.models if m.group=='gmsc_probit')
    # Repository baseline contains evidence-linked R logic; parity runs against independently executed R.
    actor='Codex technical review — team human sign-off still required'
    gate1=discover_review(output/'discover',record.model_id,actor,'approve',
                          'Source provenance inspected. Unknown ownership and business scope retained; research demo only.')
    doc_state=run_document(gate1,output/'document')
    gate2=review_document(doc_state,actor,'approve','Offline inventory-based draft inspected; research demo only.',
                         'No LLM generation or self-check occurred. Source-linked metadata checked; business and regulatory claims remain unknown. Team human review required.')
    upstream={**load_redevelop_input(gate2),'document_review_path':str(gate2.resolve())}
    rebuilt=run_redevelop(upstream,data,reference,output/'redevelop','reestimation',offline=True,baseline='gmsc_probit')
    if rebuilt['status']!='pending_redevelop_review':raise ValueError('Baseline reconstruction did not pass parity')
    gate3=review_redevelop(Path(rebuilt['run_dir'])/'state.json',actor,'approve',
                           'Hand-written IR and implementation inspected; tests and R-reference parity passed. Not an LLM conversion claim; team human review required.')
    validation=run_validation(gate3,output/'validate')
    page=collect(validation,Path('outputs/demo'))
    atomic_json(output/'latest.json',{'discover_review':str(gate1.resolve()),'document_review':str(gate2.resolve()),
                                    'redevelop_review':str(gate3.resolve()),'validation_state':str(validation.resolve()),
                                    'dashboard':str(page.resolve()),'human_signoff':'pending — technical reviews performed by Codex'})
    print(page.resolve())
    return validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/GiveMeSomeCredit'))
    p.add_argument('--reference',type=Path,default=Path('data/reference/gmsc_probit'))
    p.add_argument('--output',type=Path,default=Path('artifacts/offline_demo'))
    a=p.parse_args();run(a.data,a.reference,a.output)

if __name__=='__main__':main()
