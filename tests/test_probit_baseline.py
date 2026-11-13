import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.redevelop.baselines.gmsc_probit import fit_glm_probit_like_r, prepare, r_as_character, run
from src.redevelop.parity import compare, load_coefficients, load_predictions

ROOT = Path(__file__).parents[1]
IR = ROOT / 'src' / 'redevelop' / 'baselines' / 'gmsc_probit_ir.json'
REFERENCE = ROOT / 'data' / 'reference' / 'gmsc_probit'
DATA = ROOT / 'data' / 'GiveMeSomeCredit'


def walk_evidence(node):
    if isinstance(node, dict):
        yield from node.get('evidence', [])
        for value in node.values():
            yield from walk_evidence(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_evidence(value)


class ProbitBaselineTests(unittest.TestCase):
    def test_ir_evidence_quotes_exist_in_source(self):
        ir = json.loads(IR.read_text(encoding='utf-8'))
        lines = (ROOT / 'legacy_models' / ir['source_id']).read_text(encoding='utf-8').splitlines()
        evidence = list(walk_evidence(ir))
        self.assertGreater(len(evidence), 8)
        for item in evidence:
            excerpt = '\n'.join(lines[item['line_start'] - 1:item['line_end']])
            self.assertIn(item['quote'], excerpt, item)

    def test_r_string_comparison_and_filter_order(self):
        self.assertEqual([r_as_character(v) for v in (0.0, 98.0, 1.5)], ['0', '98', '1.5'])
        raw = pd.DataFrame({'row_id': [1, 2, 3, 4, 5], 'age': [0, 30, 30, 30, 30],
                            'NumberOfTimes90DaysLate': [0, 98, 17, 0, 0], 'DebtRatio': [0.5, 0.5, 0.5, 2.0, 0.5],
                            'RevolvingUtilizationOfUnsecuredLines': [0.1] * 5,
                            'NumberOfDependents': [None] * 5, 'MonthlyIncome': [100.0, 900.0, None, 1.0, 300.0]})
        kept = prepare(raw)
        self.assertEqual(list(kept['row_id']), [3, 5])  # age 0, code 98 and DebtRatio > 1 removed
        self.assertEqual(list(kept['MonthlyIncome']), [300.0, 300.0])  # median of the kept rows only
        self.assertEqual(list(kept['NumberOfDependents']), [0.0, 0.0])

    def test_irls_recovers_known_probit_coefficients(self):
        rng = np.random.default_rng(0)
        x = np.column_stack([np.ones(20000), rng.normal(size=(20000, 2))])
        truth = np.array([-1.0, 0.8, -0.5])
        y = (rng.uniform(size=20000) < norm.cdf(x @ truth)).astype(float)
        np.testing.assert_allclose(fit_glm_probit_like_r(x, y), truth, atol=0.05)

    @unittest.skipUnless((REFERENCE / 'coefficients.csv').exists() and (DATA / 'cs-training.csv').exists(),
                         'Needs local GMSC data and the R reference export (data/ is not in git)')
    def test_matches_r_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            counts = run(DATA, REFERENCE, out)
            self.assertEqual(counts['holdout_rows'], 33607)
            reference, labels = load_predictions(REFERENCE / 'holdout_predictions.csv')
            candidate, _ = load_predictions(out / 'holdout_predictions.csv')
            result = compare(reference, candidate, labels, 'reestimation',
                             (load_coefficients(REFERENCE / 'coefficients.csv'),
                              load_coefficients(out / 'coefficients.csv')))
        self.assertTrue(result['passed'], result['checks'])
        self.assertLess(result['metrics']['max_abs_diff'], 1e-10)


if __name__ == '__main__':
    unittest.main()
