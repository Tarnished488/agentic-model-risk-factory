"""Train original R workflows and export portable, non-executable inference references.

Models/row-level data are local only. R packages must already be installed; no notebook install cells run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

GROUPS = ('gmsc_fastwalker','gmsc_lightgbm','german_xgboost')


def fetch_fastwalker():
    root=Path('legacy_models/gmsc_fastwalker');root.mkdir(parents=True,exist_ok=True)
    url='https://www.kaggle.com/api/v1/kernels/pull/fastwalker/give-me-credit'
    data=json.load(urllib.request.urlopen(url,timeout=90))
    if data['metadata']['currentVersionNumber'] != 13:
        raise ValueError('Fastwalker source version changed; inspect before adapting')
    notebook=json.loads(data['blob']['source'])
    for cell in notebook['cells']:
        text=''.join(cell['source'])
        if '\n' not in text:text=text.replace('\\n','\n')
        cell['source']=text.splitlines(keepends=True)
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    (root/'original.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')


def prepare(group, data_dir, reference):
    source=Path('legacy_models')/group/'original.ipynb'
    if group=='gmsc_fastwalker' and not source.is_file():fetch_fastwalker()
    notebook=json.loads(source.read_text(encoding='utf-8'))
    def cell(i):return ''.join(notebook['cells'][i-1]['source'])
    # JSON string literals also safely quote these filesystem strings in R.
    code=f'data_dir <- {json.dumps(str(data_dir.resolve()))}\nout <- {json.dumps(str(reference.resolve()))}\ngroup <- "{group}"\ndir.create(out,recursive=TRUE,showWarnings=FALSE)\noptions(warn=1)\n'
    if group=='gmsc_fastwalker':
        code+='library(plyr)\nlibrary(tidyverse)\nlibrary(recipes)\nlibrary(caret)\nlibrary(gbm)\nforeach::registerDoSEQ()\n'
        for i in [3,6,7,9,10,11,13,14,17]:
            text=cell(i)
            text=text.replace("'../input/cs-training.csv'","file.path(data_dir,'cs-training.csv')").replace("'../input/cs-test.csv'","file.path(data_dir,'cs-test.csv')")
            text=text.replace('rename(Id = X1)','rename(Id = 1)').replace('step_meanimpute(', 'step_impute_mean(').replace('add_role(', 'update_role(')
            text=text.replace('cl <- makePSOCKcluster(5)\nregisterDoParallel(cl)','foreach::registerDoSEQ()').replace('stopCluster(cl)','')
            code+=text+'\n'
    elif group=='gmsc_lightgbm':
        code+='library(dplyr)\nlibrary(lightgbm)\n'
        for c in notebook['cells']:
            if c['cell_type']!='code':continue
            text=''.join(c['source'])
            text=text.replace('TrainingData <- read.csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")',
                  'full <- read.csv(file.path(data_dir,"cs-training.csv"))\nset.seed(123)\ntrain_idx <- sample(seq_len(nrow(full)),floor(.7*nrow(full)))\nTrainingData <- full[train_idx,]')
            text=text.replace('TestData <- read.csv("/kaggle/input/GiveMeSomeCredit/cs-test.csv")','TestData <- full[-train_idx,]')
            text=text.replace('"/kaggle/working/LGBM_Summit.csv"','file.path(out,"submission.csv")')
            code+=text+'\n'
    else:
        code+='library(tidyverse)\nlibrary(tidymodels)\nlibrary(themis)\nlibrary(ranger)\nlibrary(xgboost)\nlibrary(glmnet)\nset.seed(42)\n'
        for i in [10,12,33,35,36,37,39,40,41,42,43,45]:
            text=cell(i).replace('"/kaggle/input/datasets/kabure/german-credit-data-with-risk/german_credit_data.csv"','file.path(data_dir,"german_credit_data.csv")')
            code+=text+'\n'
    code+=Path('scripts/export_scoring_references.R').read_text(encoding='utf-8')
    reference.mkdir(parents=True,exist_ok=True)
    script=reference/'training_export.R';script.write_text(code,encoding='utf-8')
    env={**os.environ,'OMP_NUM_THREADS':'2'}
    with (reference/'training.log').open('w') as log:
        subprocess.run(['Rscript',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    files=['preprocessing.json','model.txt' if group=='gmsc_lightgbm' else 'model.json','train_ids.csv','holdout_predictions.csv']
    manifest={'group':group,'mode':'inference','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'data_sha256':hashlib.sha256((data_dir/('german_credit_data.csv' if group=='german_xgboost' else 'cs-training.csv')).read_bytes()).hexdigest(),
              'files':{name:hashlib.sha256((reference/name).read_bytes()).hexdigest() for name in files}}
    (reference/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=GROUPS,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.model,a.data,a.reference),indent=2))

if __name__=='__main__':main()
