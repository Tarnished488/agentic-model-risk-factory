import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.discover.discover_agent import run_discovery
from src.discover.discover_review import review as review_discover
from src.document.document_agent import run_document
from src.document.document_review import load_redevelop_input, review_document
from src.redevelop.baselines import gmsc_probit
from src.redevelop.redevelop_agent import (
    BASELINES, IR, CheckError, check_code, check_ir, difference_report, ir_differences, run_redevelop,
)
from src.redevelop.redevelop_review import load_validate_input, review_redevelop

ROOT = Path(__file__).parents[1]
BASELINE_IR = (BASELINES / 'gmsc_probit_ir.json').read_text(encoding='utf-8')
BASELINE_CODE = (BASELINES / 'gmsc_probit.py').read_text(encoding='utf-8')
BASELINE_TESTS = (BASELINES / 'gmsc_probit_test.py').read_text(encoding='utf-8')
CRASHING = ('import pandas as pd\nclass Model:\n    def fit(self, f): return self\n'
            '    def predict_proba(self, f): return None\n    def get_coefficients(self): return {}\n'
            'def run(data_dir, reference_dir, output):\n    raise RuntimeError("boom")\n')


def synthetic_gmsc(path: Path, rows: int = 3000) -> None:
    rng = np.random.default_rng(1)
    late = rng.choice([0, 0, 0, 1, 2, 98], rows)
    frame = pd.DataFrame({
        '': np.arange(1, rows + 1),
        'SeriousDlqin2yrs': 0,
        'RevolvingUtilizationOfUnsecuredLines': rng.uniform(0, 1.2, rows),
        'age': rng.integers(0, 90, rows),
        'NumberOfTime30-59DaysPastDueNotWorse': rng.integers(0, 4, rows),
        'DebtRatio': rng.uniform(0, 1.3, rows),
        'MonthlyIncome': np.where(rng.uniform(size=rows) < 0.2, np.nan, rng.uniform(1000, 9000, rows)),
        'NumberOfOpenCreditLinesAndLoans': rng.integers(0, 10, rows),
        'NumberOfTimes90DaysLate': late,
        'NumberRealEstateLoansOrLines': rng.integers(0, 3, rows),
        'NumberOfTime60-89DaysPastDueNotWorse': rng.integers(0, 3, rows),
        'NumberOfDependents': np.where(rng.uniform(size=rows) < 0.1, np.nan, rng.integers(0, 4, rows)),
    })
    signal = -1.2 + 1.5 * frame['RevolvingUtilizationOfUnsecuredLines'] + 0.4 * np.minimum(late, 3)
    frame['SeriousDlqin2yrs'] = (rng.uniform(size=rows) < 1 / (1 + np.exp(-signal))).astype(int)
    frame.to_csv(path, index=False)


def code(text):
    return json.dumps({'code': text})


@patch('src.redevelop.redevelop_agent.time.sleep')
class RedevelopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.base = Path(temp.name)
        cls.discover = cls.base / 'discover'
        inventory = run_discovery(ROOT / 'legacy_models', cls.discover)
        cls.record = next(m for m in inventory.models if m.group == 'gmsc_probit')
        cls.sources = {s.source_id: s.text for s in cls.record.sources if s.kind == 'code'}
        # Synthetic data plus a "legacy" reference produced by the trusted baseline stands in for R.
        cls.data = cls.base / 'data'
        cls.data.mkdir()
        synthetic_gmsc(cls.data / 'cs-training.csv')
        cls.reference = cls.base / 'reference'
        cls.reference.mkdir()
        kept = gmsc_probit.prepare(gmsc_probit.load(cls.data / 'cs-training.csv'))
        pd.DataFrame({'row_id': kept['row_id'].sample(frac=0.7, random_state=0)}).to_csv(
            cls.reference / 'train_ids.csv', index=False)
        gmsc_probit.run(cls.data, cls.reference, cls.reference)

    def state(self):
        return {'model_id': self.record.model_id, 'inventory_record': self.record.model_dump(),
                'document_text': '# Approved document', 'approvals': [{'gate': 'discover'}]}

    def run_agent(self, name, **kwargs):
        return run_redevelop(self.state(), self.data, self.reference, self.base / name, 'reestimation',
                             baseline='gmsc_probit', **kwargs)

    def test_baseline_ir_is_valid_and_evidenced(self, _):
        ir = IR.model_validate_json(BASELINE_IR)
        check_ir(ir, self.sources, self.record.model_id)
        self.assertEqual(ir_differences(ir, ir), [])

    def test_ir_with_fabricated_quote_or_changed_logic_is_caught(self, _):
        payload = json.loads(BASELINE_IR)
        payload['filters'][1]['evidence'][0]['quote'] = 'filter(x < 1)'
        with self.assertRaisesRegex(CheckError, 'quote mismatch'):
            check_ir(IR.model_validate(payload), self.sources, self.record.model_id)
        payload = json.loads(BASELINE_IR)
        payload['filters'][1]['compare_as'] = 'number'
        differences = ir_differences(IR.model_validate(payload), IR.model_validate_json(BASELINE_IR))
        self.assertEqual(len(differences), 1)
        self.assertIn('filters', differences[0])

    def test_static_check_blocks_dangerous_code_and_enforces_interface(self, _):
        check_code(BASELINE_CODE)
        check_code(BASELINE_TESTS, 'tests')
        for text, message in [('import os\ndef run(a, b, c): pass', 'import'),
                              ('import subprocess\ndef run(a, b, c): pass', 'import'),
                              ('def run(a, b, c): eval("1")', 'eval'),
                              ('def run(a, b, c): open("x", "w")', 'open'),
                              ('def run(a, b, c): ().__class__', 'dunder'),
                              ('x = 1', r'run\('),
                              ('def run(a, b, c): pass', 'class Model')]:
            with self.assertRaisesRegex(CheckError, message):
                check_code(text)
        with self.assertRaisesRegex(CheckError, 'import'):
            check_code('import os\ndef test_x(): pass', 'tests')

    def test_offline_mode_runs_tests_and_passes_parity(self, _):
        final = self.run_agent('offline', offline=True)
        self.assertEqual(final['status'], 'pending_redevelop_review')
        self.assertEqual(final['retry_count'], 0)
        self.assertLess(final['parity_result']['metrics']['coef_max_abs_diff'], 1e-10)
        self.assertEqual(Path(final['tests_path']).read_text(encoding='utf-8'), BASELINE_TESTS)

    def test_llm_loop_rejects_bad_tests_and_repairs_crash(self, _):
        answers = iter([
            json.dumps({**json.loads(BASELINE_IR), 'filters': []}),  # IR: valid but drops filters
            code('def test_wrong():\n    assert False\n'),           # tests fail on the baseline: rejected
            code(BASELINE_TESTS),                                     # tests accepted
            code('import os\ndef run(a, b, c): pass'),               # code rejected statically
            code(CRASHING),                                           # round 1: unit tests crash
            code(BASELINE_CODE),                                      # round 2: fixed
        ])
        seen = []

        def caller(messages, schema):
            seen.append(messages)
            return next(answers)

        with patch.dict(os.environ, {'LLM_API_KEY': 'secret-test-key', 'LLM_MODEL': 'test-model'}):
            final = self.run_agent('llm', caller=caller)
        self.assertEqual(final['status'], 'pending_redevelop_review')
        self.assertEqual(final['retry_count'], 1)
        self.assertTrue(final['ir_differences_from_baseline'])  # dropped filters flagged for gate 3
        self.assertIn('reviewed reference implementation', seen[2][-1]['content'])
        history = json.loads(Path(final['parity_result_path']).read_text(encoding='utf-8'))['history']
        self.assertIn('Generated tests failed', history[0]['error'])
        self.assertIn('boom', json.loads(seen[-1][1]['content'])['feedback'])
        self.assertNotIn('secret-test-key', json.dumps(seen))
        payload = json.loads(seen[0][1]['content'])
        self.assertEqual(list(payload['data_files']), ['cs-training.csv'])  # headers only, no rows
        self.assertTrue(any(d['source_id'].endswith('DATA_README.md') for d in payload['reference_documents']))
        audit = (Path(final['run_dir']) / 'audit_log.jsonl').read_text(encoding='utf-8')
        self.assertIn('"provider_model": "test-model"', audit)

    def test_ir_aliases_are_sent_back_and_runtime_notes_reach_codegen(self, _):
        aliased = json.loads(BASELINE_IR)
        aliased['filters'][1]['column'] = '3MonthsDefault'  # the R alias, as DeepSeek wrote it
        answers = iter([json.dumps(aliased), BASELINE_IR, code(BASELINE_TESTS), code(BASELINE_CODE)])
        seen = []

        def caller(messages, schema):
            seen.append(messages)
            return next(answers)

        final = self.run_agent('aliases', caller=caller)
        self.assertEqual(final['status'], 'pending_redevelop_review')
        retry = seen[1][-1]['content']
        self.assertIn("'3MonthsDefault'", retry)
        self.assertIn('3MonthsDefault -> NumberOfTimes90DaysLate', retry)
        notes = json.loads(seen[-1][1]['content'])['legacy_runtime_notes']
        self.assertTrue(notes and 'glm.fit' in notes[0])

    def test_wrong_model_escalates_with_difference_report(self, _):
        wrong = BASELINE_CODE.replace('epsilon: float = 1e-8', 'epsilon: float = 1e-2')
        answers = iter([BASELINE_IR, code(BASELINE_TESTS)] + [code(wrong)] * 4)
        final = self.run_agent('wrong', caller=lambda m, s: next(answers))
        self.assertEqual(final['status'], 'needs_attention')
        self.assertEqual(final['retry_count'], 3)
        report = Path(final['difference_report_path']).read_text(encoding='utf-8')
        self.assertIn('4 rounds failed', report)
        self.assertIn('Coefficient differences', report)
        with self.assertRaisesRegex(ValueError, 'passed parity'):
            review_redevelop(Path(final['run_dir']) / 'state.json', 'Reviewer', 'approve', 'Checked')

    def test_difference_report_lists_errors(self, _):
        report = difference_report('M', [{'round': 1, 'passed': False, 'error': 'Generated code failed:\nboom'}])
        self.assertIn('| 1 | Generated code failed: |', report)
        self.assertIn('boom', report)

    def test_gate3_rules_and_tamper_detection(self, _):
        final = self.run_agent('gate', offline=True)
        state_path = Path(final['run_dir']) / 'state.json'
        with self.assertRaisesRegex(ValueError, 'reviewer'):
            review_redevelop(state_path, '', 'approve', 'Checked')
        approved = review_redevelop(state_path, 'Reviewer', 'approve', 'Compared IR, code and parity')
        with self.assertRaisesRegex(ValueError, 'lineage'):
            load_validate_input(approved)  # no gate-2 snapshot behind this direct run
        Path(final['generated_code_path']).write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed'):
            review_redevelop(state_path, 'Reviewer', 'approve', 'Checked')

    def test_gate3_needs_resolution_when_ir_differs(self, _):
        answers = iter([json.dumps({**json.loads(BASELINE_IR), 'filters': []}), code(BASELINE_TESTS),
                        code(BASELINE_CODE)])
        final = self.run_agent('diff', caller=lambda m, s: next(answers))
        state_path = Path(final['run_dir']) / 'state.json'
        with self.assertRaisesRegex(ValueError, 'resolution'):
            review_redevelop(state_path, 'Reviewer', 'approve', 'Checked')
        review_redevelop(state_path, 'Reviewer', 'approve', 'Checked',
                         'Generated IR omitted filters; generated code applies them, parity confirms')

    def test_full_chain_discover_document_redevelop_gate3(self, _):
        output = self.base / 'chain'
        gate1 = review_discover(self.discover, self.record.model_id, 'Reviewer', 'approve', 'Sources checked')
        state = json.loads(run_document(gate1, output / 'document').read_text(encoding='utf-8'))
        gate2 = review_document(Path(state['doc_path']).parents[1] / 'state.json', 'Reviewer', 'approve',
                                'Document checked', 'Offline draft checked against the R source')
        upstream = {**load_redevelop_input(gate2), 'document_review_path': str(gate2.resolve())}
        final = run_redevelop(upstream, self.data, self.reference, output / 'redevelop', 'reestimation',
                              offline=True, baseline='gmsc_probit')
        gate3 = review_redevelop(Path(final['run_dir']) / 'state.json', 'Validator', 'approve', 'Parity passed')
        result = load_validate_input(gate3)
        self.assertEqual(result['status'], 'ready_for_validate')
        self.assertEqual(result['registry_stage'], 'Staging')
        self.assertEqual([a.get('gate', 'discover') for a in result['approvals']][-2:], ['document', 'redevelop'])
        self.assertEqual(len(result['approvals']), 3)
        from src.validate.validate_agent import run_validation
        from scripts.evidence_pack import collect
        validation = run_validation(gate3, output / 'validate')
        validation_state = json.loads(validation.read_text(encoding='utf-8'))
        self.assertIn(validation_state['status'], ('pending_validation_review', 'needs_attention'))
        page = collect(validation, output / 'public_evidence')
        self.assertTrue(page.is_file())
        exported = (page.parent / 'summary.json').read_text(encoding='utf-8')
        self.assertNotIn('worst_rows', exported)
        self.assertNotIn(str(self.base), exported)
        self.assertNotIn('"reviewer"', exported)
        # Validate gets everything it needs to recompute metrics, all hash-locked.
        self.assertEqual(Path(result['data_path']), (self.data / 'cs-training.csv').resolve())
        self.assertEqual(Path(result['train_ids_path']), (self.reference / 'train_ids.csv').resolve())
        predictions = Path(result['holdout_predictions_path'])
        self.assertEqual(list(pd.read_csv(predictions).columns), ['row_id', 'probability', 'label'])
        self.assertTrue(Path(result['coefficients_path']).exists())
        original = predictions.read_text(encoding='utf-8')
        predictions.write_text(original.replace('0.', '0.9', 1), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'holdout_predictions_path changed'):
            load_validate_input(gate3)
        predictions.write_text(original, encoding='utf-8')
        load_validate_input(gate3)
        Path(result['generated_code_path']).write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_validate_input(gate3)

    def test_mlflow_logs_parent_and_nested_rounds(self, _):
        import mlflow
        uri = (self.base / 'mlruns').as_uri()
        with patch.dict(os.environ, {'MLFLOW_TRACKING_URI': uri, 'MLFLOW_ALLOW_FILE_STORE': 'true'}):
            mlflow.set_tracking_uri(uri)
            final = self.run_agent('mlflow', offline=True, use_mlflow=True)
            runs = mlflow.search_runs(search_all_experiments=True)
        self.assertEqual(len(runs), 2)  # parent + one round
        self.assertEqual(set(runs['tags.mlflow.runName']), {'redevelop-' + final['redevelop_run_id'], 'round-1'})

    def test_mlflow_failure_does_not_lose_the_run(self, _):
        with patch('src.redevelop.redevelop_agent.log_redevelop_mlflow', side_effect=RuntimeError('offline')):
            final = self.run_agent('mlflow_down', offline=True, use_mlflow=True)
        self.assertEqual(final['status'], 'pending_redevelop_review')
        self.assertIn('mlflow_failed', (Path(final['run_dir']) / 'audit_log.jsonl').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
