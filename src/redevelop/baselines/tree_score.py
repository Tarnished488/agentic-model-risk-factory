"""Portable inference migration: Python preprocessing + exported R learned parameters.

No R runtime, RDS or pickle is used by this module. This is not Python re-training.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit


def lightgbm_features(raw):
    frame = raw.rename(columns=lambda x: x.replace('-', '.')).copy()
    frame = frame.drop(columns=['row_id', 'SeriousDlqin2yrs'], errors='ignore')
    frame['TotalPastDue'] = (frame['NumberOfTime30.59DaysPastDueNotWorse'] +
                            frame['NumberOfTime60.89DaysPastDueNotWorse'] + frame['NumberOfTimes90DaysLate'])
    frame['DebtRatio1000'] = frame['DebtRatio'].where(frame['MonthlyIncome'].notna() & (frame['MonthlyIncome'] >= 1000))
    names = list(frame)
    result = {k: frame[k].to_numpy(float) for k in names}
    for name in names:
        a = frame[name].to_numpy(float)
        mean = frame[name].mean()
        result[name+'_minus_mean'] = a - mean
        if mean != 0:
            result[name+'_divided_by_mean'] = a / mean
        with np.errstate(divide='ignore', invalid='ignore'):
            result['log_'+name] = np.where(a == 0, np.nan, np.log(a))
        result[name+'_squared'] = a*a
    for i, name in enumerate(names):
        for other in names[i+1:]:
            a, b = result[name], result[other]
            result[name+'_times_'+other] = a*b
            result[name+'_minus_'+other] = a-b
            with np.errstate(divide='ignore', invalid='ignore'):
                result[name+'_divided_by_'+other] = np.where((b == 0) | np.isnan(b), np.nan, a/b)
            result[name+'_plus_'+other] = a+b
    result['(Intercept)'] = np.ones(len(frame))
    return pd.DataFrame(result, index=raw.index)


def gbm_features(raw, config):
    frame = raw.rename(columns=lambda x: x.lower()).copy()
    values = {}
    for feature in config['feature_order']:
        name = feature.strip('`')
        a = pd.to_numeric(frame[name], errors='raise').fillna(config['impute'][name])
        values[feature] = (a - config['center'][name]) / config['scale'][name]
    return pd.DataFrame(values, index=raw.index)


def german_features(raw, config):
    f = raw.rename(columns=lambda x: x.lower().replace(' ', '_').replace('.', '_')).copy()
    for name in ['saving_accounts', 'checking_account']:
        f[name] = f[name].fillna('unknown')
    f['job'] = f['job'].astype(int).map({0:'unskilled non-resident',1:'unskilled resident',2:'skilled',3:'highly skilled'})
    f['housing'] = f['housing'].str.lower()
    f['monthly_payment'] = f['credit_amount'] / f['duration']
    f['high_loan_flag'] = (f['credit_amount'] > config['high_loan_cutoff']).astype(float)
    f['long_loan_flag'] = (f['duration'] > 24).astype(float)
    f['amount_x_duration'] = f['credit_amount']*f['duration']
    f['young_borrower'] = (f['age'] < 30).astype(float)
    f['senior_borrower'] = (f['age'] >= 50).astype(float)
    f['skilled_worker'] = f['job'].isin(['skilled','highly skilled']).astype(float)
    f['owns_home'] = (f['housing'] == 'own').astype(float)
    for name, mapping in config['categories'].items():
        if not f[name].isin(mapping['levels']).all():
            raise ValueError('Unseen or missing category: '+name)
        for level, column in zip(mapping['levels'][1:], mapping['columns']):
            f[column] = (f[name] == level).astype(float)
    columns = config['feature_order']
    return pd.DataFrame({k:(f[k]-config['center'][k])/config['scale'][k] for k in columns},index=raw.index)


class Model:
    def __init__(self, reference_dir=None):
        if reference_dir is not None:
            self.reference_dir = Path(reference_dir)
            self.config = json.loads((self.reference_dir/'preprocessing.json').read_text(encoding='utf-8'))
            for key in ['impute','center','scale']:
                if key in self.config:
                    self.config[key] = {k:float(v) for k,v in self.config[key].items()}
            kind = self.config['kind']
            if kind == 'lightgbm':
                import lightgbm
                self.engine = lightgbm.Booster(model_file=str(self.reference_dir/'model.txt'))
            elif kind == 'xgboost':
                import xgboost
                self.engine = xgboost.Booster(model_file=str(self.reference_dir/'model.json'))
                self.engine.set_param({'nthread':2})
            elif kind == 'gbm':
                self.engine = json.loads((self.reference_dir/'model.json').read_text(encoding='utf-8'))
            else:
                raise ValueError('Unsupported portable model kind')

    def fit(self, frame):
        raise NotImplementedError('Inference migration loads R learned parameters; Python re-training is not implemented')

    def prepare(self, raw):
        kind = self.config['kind']
        if kind == 'lightgbm':
            frame = lightgbm_features(raw).loc[:,self.config['feature_order']]
        elif kind == 'xgboost':
            frame = german_features(raw,self.config)
        else:
            frame = gbm_features(raw,self.config)
        return frame

    def predict_proba(self, raw):
        frame = self.prepare(raw)
        kind = self.config['kind']
        if kind == 'lightgbm':
            probabilities = self.engine.predict(frame.to_numpy(),num_threads=2)
        elif kind == 'xgboost':
            import xgboost
            probabilities = self.engine.predict(xgboost.DMatrix(frame.to_numpy(),feature_names=list(frame)))
        else:
            x = frame.to_numpy(float)
            eta = np.full(len(x), self.engine['initF'], dtype=float)
            for tree in self.engine['trees']:
                sv = np.asarray(tree['split_var'], dtype=int)
                code = np.asarray(tree['code'], dtype=float)
                left,right,missing = [np.asarray(tree[k],dtype=int) for k in ['left','right','missing']]
                node = np.zeros(len(x),dtype=int)
                for _ in range(len(sv)):
                    active = np.flatnonzero(sv[node] >= 0)
                    if not len(active):break
                    n = node[active]
                    value = x[active,sv[n]]
                    node[active] = np.where(np.isnan(value),missing[n],np.where(value < code[n],left[n],right[n]))
                if np.any(sv[node] >= 0):raise ValueError('Invalid exported tree: cycle')
                eta += np.asarray(tree['prediction'],dtype=float)[node]
            probabilities = expit(eta)
        if self.config.get('complement_bad_probability',False):
            probabilities = 1-probabilities
        return np.asarray(probabilities,dtype=float)

    def get_coefficients(self):
        return {}  # Tree models have no linear coefficients; Validate must not invent them.


def run(data_dir, reference_dir, output):
    model = Model(reference_dir)
    config = model.config
    raw = pd.read_csv(Path(data_dir)/config['data_file'])
    raw = raw.rename(columns={raw.columns[0]:'row_id'})
    if raw.row_id.isna().any() or raw.row_id.duplicated().any():raise ValueError('Invalid row IDs')
    train_ids = set(pd.read_csv(Path(reference_dir)/'train_ids.csv').row_id)
    if not train_ids or not train_ids <= set(raw.row_id):raise ValueError('Training IDs absent from data')
    holdout = raw[~raw.row_id.isin(train_ids)].copy()
    labels = holdout[config['target']]
    if config.get('target_mapping'):
        labels = labels.map(config['target_mapping'])
    if labels.isna().any() or not set(labels) <= {0,1}:raise ValueError('Invalid target mapping')
    predictions = model.predict_proba(holdout)
    if not np.isfinite(predictions).all() or ((predictions<0)|(predictions>1)).any():raise ValueError('Invalid predictions')
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row_id':holdout.row_id,'probability':predictions,'label':labels.astype(int)}).to_csv(output/'holdout_predictions.csv',index=False)
    return {'train_rows':len(train_ids),'holdout_rows':len(holdout),'filtered_rows':len(raw),'migration_mode':'inference; R learned parameters, Python preprocessing and scoring'}
