"""Run the three reviewed scoring migrations in an isolated artifact tree.

Use --technical-rehearsal to record explicitly AI technical checks at gates 1–3.
Gate 4 is never automatically approved. Ordinary human use: src.pipeline / Streamlit.
"""
import argparse
import json
import shutil
from pathlib import Path
import src.pipeline as pipeline
from src.discover.discover_agent import atomic_json, file_sha256

GROUPS=('gmsc_fastwalker','gmsc_lightgbm','german_xgboost')


def run(output:Path,public:Path,technical=False):
    pipeline.ARTIFACTS=output
    results={}
    for group in GROUPS:
        while True:
            kind,step=pipeline.run(group,False)
            state=pipeline.load_state(group)
            if kind!='await' or step=='gate4' or not technical:break
            if step=='gate3' and state['steps']['redevelop']['status']!='pending_redevelop_review':break
            pipeline.review(group,'Codex technical rehearsal — team human review pending','approve',
                'Offline scoring-migration technical checks only. Not team human sign-off.',
                'Preserved R learned parameters; Python preprocessing and scoring checked against R. Original CV/preprocessing limitations retained for review. No Python re-training or LLM conversion claim.')
        state=pipeline.load_state(group)
        result={'model_id':state['model_id'],'next_action':[kind,step],'error':state['last_error'],
                'scope':'inference migration; Python preprocessing/scoring, R learned parameters; not LLM generated',
                'human_review':'pending — rehearsal gates are explicitly technical checks'}
        if 'redevelop' in state['steps']:
            rs=json.loads(Path(state['steps']['redevelop']['state_path']).read_text())
            parity=rs['parity_result']
            result['parity']={k:v for k,v in parity.items() if k!='worst_rows'}
            result['runtime_asset_hashes']={Path(k).name:v for k,v in rs.get('runtime_assets_sha256',{}).items()}
        if 'validate' in state['steps']:
            vs=json.loads(Path(state['steps']['validate']['state_path']).read_text())
            report=json.loads(Path(vs['metrics_path']).read_text())
            result['validation']=report
            dest=public/group;dest.mkdir(parents=True,exist_ok=True)
            for key,name in [('html_path','report.html'),('report_path','report.md'),('findings_path','findings.json')]:
                shutil.copy2(vs[key],dest/name)
            result['report']='./'+group+'/report.html'
        results[group]=result
        public.mkdir(parents=True,exist_ok=True)
        atomic_json(public/'summary.json',results)
        atomic_json(output/'summary.json',results)
        print(group,kind,step,result.get('validation',{}).get('overall_rating'),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/three_rebuild/pipeline'))
    p.add_argument('--public',type=Path,default=Path('outputs/three_rebuild'))
    p.add_argument('--technical-rehearsal',action='store_true')
    a=p.parse_args();run(a.output,a.public,a.technical_rehearsal)

if __name__=='__main__':main()
