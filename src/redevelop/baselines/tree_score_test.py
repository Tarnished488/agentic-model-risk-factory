"""Synthetic inference contracts copied next to the candidate by Redevelop."""
import json
import numpy as np
import pandas as pd
import pytest
import new_model


def test_inference_does_not_claim_retraining():
    with pytest.raises(NotImplementedError):new_model.Model().fit(pd.DataFrame())


def test_portable_tree_missing_and_split_direction(tmp_path):
    config={'kind':'gbm','feature_order':['age'],'impute':{'age':40},'center':{'age':40},'scale':{'age':10}}
    (tmp_path/'preprocessing.json').write_text(json.dumps(config))
    # Children are zero based; leaves already include shrinkage.
    tree={'split_var':[0,-1,-1,-1],'code':[0,0,0,0],'left':[1,-1,-1,-1],'right':[2,-1,-1,-1],
          'missing':[3,-1,-1,-1],'prediction':[0,-1,1,.25]}
    (tmp_path/'model.json').write_text(json.dumps({'initF':0,'trees':[tree]}))
    model=new_model.Model(tmp_path)
    result=model.predict_proba(pd.DataFrame({'age':[30.,40.,None]}))
    np.testing.assert_allclose(result,[1/(1+np.exp(1)),1/(1+np.exp(-1)),1/(1+np.exp(-1))])
    assert model.get_coefficients()=={}


def test_german_unseen_category_is_rejected():
    raw=pd.DataFrame({'Age':[30],'Job':[2],'Housing':['unknown-new'],'Saving accounts':['little'],
                      'Checking account':['little'],'Credit amount':[1000.],'Duration':[12.],'Purpose':['car']})
    config={'high_loan_cutoff':2000,'categories':{'housing':{'levels':['own','rent'],'columns':['housing_rent']}},
            'feature_order':['age'],'center':{'age':30},'scale':{'age':10}}
    with pytest.raises(ValueError,match='Unseen'):new_model.german_features(raw,config)
