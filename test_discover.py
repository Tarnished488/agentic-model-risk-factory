import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from discover_agent import (
    Evidence, Metadata, Source, extract_with_llm, load_document_input,
    offline_metadata, read_sources, review, run_discovery, validate_evidence,
)


class DiscoverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'models'
        self.group = self.root / 'loan'
        self.group.mkdir(parents=True)
        self.code = self.group / 'model.R'
        self.code.write_text('library(lightgbm)\nm <- lightgbm(data = x)\npredict(m, x)\n')
        self.output = Path(self.temp.name) / 'artifacts'

    def scan(self):
        return run_discovery(self.root, self.output)

    def test_stable_identity_and_separate_runs(self):
        a, b = self.scan(), self.scan()
        self.assertEqual(a.models[0].model_id, b.models[0].model_id)
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertTrue((self.output / 'runs' / a.run_id / 'inventory.json').exists())
        self.assertIsNone(b.models[0].risk_tier)
        self.assertIsNone(b.models[0].metadata.business_owner)
        self.assertEqual(b.models[0].metadata.doc_status, 'missing')

    def test_two_groups_do_not_merge(self):
        other = self.root / 'fraud'
        other.mkdir()
        (other / 'model.py').write_text('import math\n')
        self.assertEqual(len(self.scan().models), 2)

    def test_notebook_source_cell_positions_and_no_outputs(self):
        path = self.group / 'n.ipynb'
        path.write_text(json.dumps({'metadata': {'kernelspec': {'language': 'R'}}, 'cells': [
            {'cell_type': 'markdown', 'source': ['# Model']},
            {'cell_type': 'code', 'source': ['\nlibrary(lightgbm)\n'], 'outputs': [{'text': 'DO NOT READ'}]},
        ]}))
        sources = read_sources(path, 'n.ipynb', 'abc')
        self.assertEqual(sources[1].source_id, 'n.ipynb#cell=2')
        self.assertNotIn('DO NOT READ', sources[1].text)
        validate_evidence(offline_metadata(sources), sources)

    def test_csv_extracts_header_without_rows(self):
        path = self.group / 'data.csv'
        path.write_text('age,income\n21,PRIVATE_VALUE\n')
        sources = read_sources(path, 'data.csv', 'abc')
        self.assertNotIn('PRIVATE_VALUE', sources[0].text)
        self.assertEqual(offline_metadata(sources).inputs, ['age', 'income'])

    def test_invalid_evidence_cannot_pass(self):
        source = Source(source_id='x.R', path='x.R', kind='code', sha256='abc', text='library(lightgbm)')
        metadata = Metadata(algorithm='LightGBM', evidence={'algorithm': [
            Evidence(source_id='x.R', line_start=1, line_end=1, quote='library(xgboost)')]})
        with self.assertRaisesRegex(ValueError, 'quote mismatch'):
            validate_evidence(metadata, [source])

    def test_llm_retries_invalid_json_and_hallucinated_source(self):
        source = Source(source_id='x.R', path='x.R', kind='code', sha256='abc', text='library(lightgbm)')
        bad = Metadata(algorithm='LightGBM', evidence={'algorithm': [
            Evidence(source_id='fake.R', line_start=1, line_end=1, quote='library(lightgbm)')]})
        good = Metadata(algorithm='LightGBM', evidence={'algorithm': [
            Evidence(source_id='x.R', line_start=1, line_end=1, quote='library(lightgbm)')]})
        answers = iter(['not json', bad.model_dump_json(), good.model_dump_json()])
        calls = []
        def caller(messages, schema):
            calls.append(len(messages))
            return next(answers)
        with patch('discover_agent.time.sleep'):
            result = extract_with_llm([source], caller)
        self.assertEqual(result.algorithm, 'LightGBM')
        self.assertEqual(len(calls), 3)

    def test_llm_failure_is_explicit_and_cannot_be_approved(self):
        with patch('discover_agent.call_llm', return_value='no'):
            # extract_with_llm is independently covered; exercise the run's fallback.
            with patch('discover_agent.extract_with_llm', side_effect=ValueError('failed')):
                result = run_discovery(self.root, self.output, use_llm=True)
        self.assertEqual(result.models[0].status, 'needs_attention')
        with self.assertRaisesRegex(ValueError, 'Resolve'):
            review(self.output, result.models[0].model_id, 'Tester', 'approve', 'Checked')

    def test_approval_handoff_and_source_change(self):
        model = self.scan().models[0]
        approved = review(self.output, model.model_id, 'Tester', 'approve', '已核对来源和未知项')  # non-ASCII: Windows cp936 regression
        self.assertEqual(load_document_input(approved)['status'], 'ready_for_document')
        self.code.write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            load_document_input(approved)

    def test_input_root_is_relative_inside_working_directory(self):
        cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, cwd)
        result = run_discovery(Path('models'), Path('artifacts'))
        self.assertEqual(result.input_root, 'models')
        approved = review(Path('artifacts'), result.models[0].model_id, 'Tester', 'approve', 'Checked')
        self.assertEqual(load_document_input(approved)['input_root'], 'models')

    def test_modified_approved_record_and_rejection_are_blocked(self):
        model = self.scan().models[0]
        rejected = review(self.output, model.model_id, 'Tester', 'reject', 'Needs more information')
        with self.assertRaisesRegex(ValueError, 'approval'):
            load_document_input(rejected)
        approved = review(self.output, model.model_id, 'Tester', 'approve', 'Checked')
        payload = json.loads(approved.read_text(encoding='utf-8'))
        payload['inventory_record']['metadata']['name'] = 'tampered'
        approved.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'changed after review'):
            load_document_input(approved)

    def test_bad_notebook_preserves_other_group_and_surfaces_error(self):
        (self.group / 'broken.ipynb').write_text('{')
        result = self.scan()
        self.assertEqual(result.models[0].status, 'needs_attention')
        self.assertTrue(result.models[0].warnings)

    def test_external_symlink_is_not_read(self):
        outside = Path(self.temp.name) / 'outside.R'
        outside.write_text('OUTSIDE_SECRET')
        (self.group / 'link.R').symlink_to(outside)
        result = self.scan()
        self.assertNotIn('OUTSIDE_SECRET', result.model_dump_json())
        self.assertEqual(result.models[0].status, 'needs_attention')

    def test_output_inside_input_tree_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            run_discovery(self.root, self.group / 'results')

    def test_actual_kaggle_notebook(self):
        result = run_discovery(Path(__file__).parent / 'legacy_models', self.output)
        record = result.models[0]
        self.assertEqual(record.metadata.language, 'R')
        self.assertEqual(record.metadata.algorithm, 'LightGBM')
        self.assertEqual(record.metadata.target, 'SeriousDlqin2yrs')
        self.assertEqual(len(record.metadata.training_stages), 2)
        self.assertEqual(record.metadata.inputs, [])  # No raw dataset header supplied.
        self.assertEqual(record.status, 'pending_review')
        self.assertEqual(len(result.models), 1)


if __name__ == '__main__':
    unittest.main()
