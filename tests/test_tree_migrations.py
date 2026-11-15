import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.redevelop.baselines import tree_score
from src.redevelop.redevelop_agent import BASELINES, IR, check_code, run_tests
from src.redevelop.redevelop_review import _check_handoff_files
from src.discover.discover_agent import file_sha256


class PortableMigrationTests(unittest.TestCase):
    def test_reviewed_tree_tests_run_in_candidate_subprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            run_tests((BASELINES/'tree_score.py').read_text(),(BASELINES/'tree_score_test.py').read_text(),Path(temp))

    def test_native_engines_are_only_allowed_for_reviewed_tree_code(self):
        code=(BASELINES/'tree_score.py').read_text()
        with self.assertRaisesRegex(ValueError,'Forbidden import'):check_code(code)
        check_code(code,trusted_tree=True)
        with self.assertRaisesRegex(ValueError,'Forbidden import'):
            check_code('import subprocess\n'+code,trusted_tree=True)

    def test_tree_ir_declares_inference_and_target_semantics(self):
        for name in ['gmsc_fastwalker','gmsc_lightgbm','german_xgboost']:
            ir=IR.model_validate_json((BASELINES/(name+'_ir.json')).read_text())
            self.assertEqual(ir.migration.mode,'inference')
            self.assertIn('preprocessing.json',ir.migration.assets)
            self.assertTrue(ir.migration.limitations)
        ir=IR.model_validate_json((BASELINES/'german_xgboost_ir.json').read_text())
        self.assertEqual(ir.model.target_mapping,{'good':0,'bad':1})

    def test_changed_runtime_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'model.json';p.write_text('{}')
            state={'handoff_sha256':{},'runtime_assets_sha256':{str(p):file_sha256(p)}}
            _check_handoff_files(state)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Runtime model asset'):_check_handoff_files(state)

    def test_seventeen_digit_thresholds_preserve_branch_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            config={'kind':'gbm','feature_order':['age'],'impute':{'age':'0'},'center':{'age':'0'},'scale':{'age':'1'}}
            (p/'preprocessing.json').write_text(json.dumps(config))
            tree={'split_var':[0,-1,-1],'code':['0.33333333333333331','0','0'],
                  'left':[1,-1,-1],'right':[2,-1,-1],'missing':[1,-1,-1],'prediction':['0','-1','1']}
            (p/'model.json').write_text(json.dumps({'initF':'0','trees':[tree]}))
            model=tree_score.Model(p)
            values=pd.DataFrame({'age':[np.nextafter(1/3,0),1/3]})
            np.testing.assert_allclose(model.predict_proba(values),[1/(1+np.exp(1)),1/(1+np.exp(-1))])

    def test_lightgbm_interactions_and_missing_income(self):
        raw=pd.DataFrame({'age':[0,30], 'MonthlyIncome':[np.nan,500.], 'DebtRatio':[.1,.2],
             'NumberOfTime30-59DaysPastDueNotWorse':[1.,2.], 'NumberOfTime60-89DaysPastDueNotWorse':[2.,0.],
             'NumberOfTimes90DaysLate':[0.,1.]})
        f=tree_score.lightgbm_features(raw)
        self.assertTrue(f.DebtRatio1000.isna().all())
        self.assertTrue(np.isnan(f.log_age.iloc[0]))
        self.assertEqual(f.age.iloc[0],0)  # Original scoring does not apply the training age correction.
        np.testing.assert_array_equal(f.TotalPastDue,[3,3])
        self.assertTrue(np.isnan(f['age_divided_by_MonthlyIncome'].iloc[0]))

    def test_german_dummies_use_training_reference_and_do_not_include_sex(self):
        raw=pd.DataFrame({'Age':[30],'Sex':['female'],'Job':[2],'Housing':['own'], 'Saving accounts':[None],
             'Checking account':['little'],'Credit amount':[1200.],'Duration':[12.]})
        config={'high_loan_cutoff':1000,'categories':{'housing':{'levels':['free','own','rent'],'columns':['housing_own','housing_rent']}},
             'feature_order':['age','monthly_payment','housing_own'],
             'center':{'age':20,'monthly_payment':50,'housing_own':.5},'scale':{'age':10,'monthly_payment':25,'housing_own':.5}}
        f=tree_score.german_features(raw,config)
        np.testing.assert_allclose(f.to_numpy(),[[1,2,1]])
        self.assertNotIn('Sex',f)

    def test_real_portable_predictions_match_r_if_available(self):
        from src.redevelop.parity import compare,load_predictions
        import os
        if not os.environ.get('RUN_REAL_TREE_TESTS'):self.skipTest('Set RUN_REAL_TREE_TESTS=1 with local references and data')
        for group,reference,data in [('gmsc_fastwalker','gmsc_fastwalker','GiveMeSomeCredit'),('gmsc_lightgbm','gmsc_lightgbm_labelled','GiveMeSomeCredit'),('german_xgboost','german_xgboost','GermanCredit')]:
            r=Path('data/reference')/reference
            with tempfile.TemporaryDirectory() as temp:
                tree_score.run(Path('data')/data,r,Path(temp))
                a,y=load_predictions(r/'holdout_predictions.csv');b,_=load_predictions(Path(temp)/'holdout_predictions.csv')
                result=compare(a,b,y,'inference')
                self.assertTrue(result['passed'],(group,result['metrics']))

if __name__=='__main__':unittest.main()
