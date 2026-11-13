import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.discover.discover_agent import atomic_json, file_sha256
from src.validate.validate_agent import DEFAULT_POLICY, evaluate, performance, psi, read_policy, run_validation
from src.validate.validate_review import review_validation


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rows = 400
        frame = pd.DataFrame({'id': np.arange(rows).astype(str), 'target': np.tile([0,1], rows//2),
                              'age': np.tile([25,25,55,55], rows//4), 'feature': np.tile([0.,1.],rows//2)})
        frame.to_csv(self.root/'raw.csv',index=False)
        pd.DataFrame({'row_id':np.arange(200)}).to_csv(self.root/'train.csv',index=False)
        pd.DataFrame({'row_id':np.arange(200,400), 'probability':np.tile([.05,.95],100),
                      'label':np.tile([0,1],100)}).to_csv(self.root/'pred.csv',index=False)
        ir = {'data':{'row_id':'id'},'model':{'target':'target','features':['feature','age']},'filters':[]}
        atomic_json(self.root/'ir.json',ir)
        self.state = {'model_id':'TEST', 'status':'ready_for_validate','ir_path':str(self.root/'ir.json'),
                      'data_path':str(self.root/'raw.csv'),'train_ids_path':str(self.root/'train.csv'),
                      'holdout_predictions_path':str(self.root/'pred.csv'),'approvals':[]}
        self.policy=read_policy(DEFAULT_POLICY)

    def test_metric_hand_calculations(self):
        result=performance([0,1,0,1],[0,1,0,1])
        self.assertEqual(result['auc'],1)
        self.assertEqual(result['ks'],1)
        self.assertEqual(result['brier'],0)
        self.assertEqual(result['ece'],0)
        self.assertAlmostEqual(performance([0,1],[.5,.5])['auc'],.5)

    def test_performance_and_group_support(self):
        r=evaluate(self.state,self.policy)
        self.assertEqual(r['status'],'pending_validation_review')
        self.assertEqual(len(r['groups']),2)
        self.assertTrue(all(g['sufficient'] for g in r['groups']))
        self.assertEqual(r['metrics']['rows'],200)

    def test_overlap_rejected(self):
        p=pd.read_csv(self.state['holdout_predictions_path']);p.loc[0,'row_id']=0
        p.to_csv(self.state['holdout_predictions_path'],index=False)
        with self.assertRaisesRegex(ValueError,'leakage'):evaluate(self.state,self.policy)

    def test_forged_labels_rejected(self):
        p=pd.read_csv(self.state['holdout_predictions_path']);p.loc[0,'label']=1
        p.to_csv(self.state['holdout_predictions_path'],index=False)
        with self.assertRaisesRegex(ValueError,'labels'):evaluate(self.state,self.policy)

    def test_missing_rows_rejected(self):
        p=pd.read_csv(self.state['holdout_predictions_path']);p.iloc[1:].to_csv(self.state['holdout_predictions_path'],index=False)
        with self.assertRaisesRegex(ValueError,'eligible'):evaluate(self.state,self.policy)

    def test_duplicates_and_nonfinite_rejected(self):
        p=pd.read_csv(self.state['holdout_predictions_path']);p.loc[0,'probability']=np.inf
        p.to_csv(self.state['holdout_predictions_path'],index=False)
        with self.assertRaisesRegex(ValueError,'probabilities'):evaluate(self.state,self.policy)
        pd.concat([p,p.iloc[:1]]).to_csv(self.state['holdout_predictions_path'],index=False)
        with self.assertRaisesRegex(ValueError,'duplicate'):evaluate(self.state,self.policy)

    def test_insufficient_groups_never_pass_fairness(self):
        p=copy.deepcopy(self.policy);p['min_group_size']=1000
        r=evaluate(self.state,p)
        self.assertEqual(next(f for f in r['findings'] if f['check']=='age_group_tpr_gap')['result'],'unverifiable')

    def test_psi_detects_shift_including_missing(self):
        a=pd.Series(np.arange(100,dtype=float))
        self.assertEqual(psi(a,a),0)
        self.assertGreater(psi(a,pd.Series([1000.]*100)),.25)
        self.assertGreater(psi(a,pd.Series([None]*100)),.25)

    def test_policy_failure_blocks_approval_and_report_tampering(self):
        self.policy['min_auc']=1
        self.policy['max_ece']=0
        atomic_json(self.root/'policy.json',self.policy)
        review=self.root/'upstream.json';review.write_text('{}')
        with patch('src.validate.validate_agent.load_validate_input',return_value=self.state):
            path=run_validation(review,self.root/'output',self.root/'policy.json')
        with patch('src.validate.validate_review.load_validate_input',return_value=self.state):
            with self.assertRaisesRegex(ValueError,'Failed'):review_validation(path,'tester','approve','test','resolved')
            s=json.loads(path.read_text());Path(s['report_path']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):review_validation(path,'tester','reject','test')

    def test_gate4_requires_resolution_and_preserves_snapshot(self):
        review=self.root/'upstream.json';review.write_text('{}')
        with patch('src.validate.validate_agent.load_validate_input',return_value=self.state):
            path=run_validation(review,self.root/'output')
        with patch('src.validate.validate_review.load_validate_input',return_value=self.state):
            with self.assertRaisesRegex(ValueError,'resolution'):review_validation(path,'tester','approve','test')
            approved=review_validation(path,'tester','approve','test','research only; limitations reviewed')
            s=json.loads(approved.read_text());self.assertEqual(s['status'],'validated_for_research')
            self.assertEqual(s['approvals'][-1]['gate'],'validate')

    def test_unapproved_input_cannot_validate(self):
        review=self.root/'not-approved.json';review.write_text('{}')
        with self.assertRaises((ValueError,KeyError)):run_validation(review,self.root/'output')

if __name__=='__main__':unittest.main()
