import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.redevelop.parity import auc, compare, ks, load_predictions, record_round, spearman

REFERENCE = {'a': 0.1, 'b': 0.4, 'c': 0.35, 'd': 0.8}
LABELS = {'a': 0, 'b': 0, 'c': 1, 'd': 1}


class ParityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)

    def test_metrics_match_hand_calculation(self):
        scores, labels = list(REFERENCE.values()), list(LABELS.values())
        self.assertAlmostEqual(auc(scores, labels), 0.75)
        self.assertAlmostEqual(ks(scores, labels), 0.5)
        self.assertAlmostEqual(auc([0.5] * 4, labels), 0.5)  # ties count half
        self.assertAlmostEqual(spearman(scores, [s * 2 for s in scores]), 1.0)

    def test_inference_tier_is_exact_and_reestimation_tolerates_monotone_shift(self):
        self.assertTrue(compare(REFERENCE, dict(REFERENCE), None, 'inference')['passed'])
        shifted = {k: v + 1e-3 for k, v in REFERENCE.items()}
        self.assertFalse(compare(REFERENCE, shifted, None, 'inference')['passed'])
        self.assertTrue(compare(REFERENCE, shifted, LABELS, 'reestimation')['passed'])

    def test_bad_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Row ids differ'):
            compare(REFERENCE, {'a': 0.1}, None, 'inference')
        with self.assertRaisesRegex(ValueError, 'needs labels'):
            compare(REFERENCE, dict(REFERENCE), None, 'reestimation')
        with self.assertRaisesRegex(ValueError, r'\[0, 1\]'):
            compare(REFERENCE, {**REFERENCE, 'a': 1.5}, None, 'inference')
        path = self.base / 'dup.csv'
        path.write_text('row_id,probability\n1,0.2\n1,0.3\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_predictions(path)

    def test_coefficient_parity(self):
        coef = {'(Intercept)': -1.0, 'age': 0.01}
        result = compare(REFERENCE, dict(REFERENCE), None, 'inference', (coef, dict(coef)))
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['worst_rows']), 4)  # predictions still reported alongside coefficients
        result = compare(REFERENCE, dict(REFERENCE), None, 'inference', (coef, {**coef, 'age': 0.011}))
        self.assertFalse(result['passed'])
        with self.assertRaisesRegex(ValueError, 'Coefficient terms differ'):
            compare(REFERENCE, dict(REFERENCE), None, 'inference', (coef, {'age': 0.01}))

    def test_failures_retry_three_times_then_go_to_human(self):
        failing = compare(REFERENCE, {k: v + 0.01 for k, v in REFERENCE.items()}, None, 'inference')
        statuses = [record_round(self.base, 'M1', failing)['status'] for _ in range(4)]
        self.assertEqual(statuses, ['retry', 'retry', 'retry', 'needs_attention'])
        self.assertTrue((self.base / 'M1' / 'audit_log.jsonl').exists())
        passing = compare(REFERENCE, dict(REFERENCE), None, 'inference')
        self.assertEqual(record_round(self.base, 'M2', passing)['status'], 'pending_redevelop_review')

    def test_cli_round_trip(self):
        rows = '\n'.join(f'{k},{v},{LABELS[k]}' for k, v in REFERENCE.items())
        (self.base / 'ref.csv').write_text('row_id,probability,label\n' + rows + '\n', encoding='utf-8')
        (self.base / 'new.csv').write_text('row_id,probability,label\n' + rows + '\n', encoding='utf-8')
        result = subprocess.run([sys.executable, '-m', 'src.redevelop.parity', str(self.base / 'ref.csv'),
                                 str(self.base / 'new.csv'), '--model-id', 'M1', '--tier', 'reestimation',
                                 '--output', str(self.base / 'out')],
                                capture_output=True, text=True, cwd=Path(__file__).parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('pending_redevelop_review', result.stdout)


if __name__ == '__main__':
    unittest.main()
