import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import pipeline
from src.redevelop.baselines import gmsc_probit
from tests.test_redevelop import synthetic_gmsc


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        data, reference = base / 'data', base / 'reference'
        data.mkdir()
        reference.mkdir()
        synthetic_gmsc(data / 'cs-training.csv')
        kept = gmsc_probit.prepare(gmsc_probit.load(data / 'cs-training.csv'))
        pd.DataFrame({'row_id': kept['row_id'].sample(frac=0.7, random_state=0)}).to_csv(
            reference / 'train_ids.csv', index=False)
        gmsc_probit.run(data, reference, reference)
        # Synthetic labels ignore age and past-due counts, so their coefficient signs are random here.
        policy = json.loads(Path('config/validation_policy.json').read_text(encoding='utf-8'))
        (base / 'policy.json').write_text(json.dumps({**policy, 'expected_signs': {}}), encoding='utf-8')
        config = base / 'pipeline.json'
        config.write_text(json.dumps({'legacy_root': 'legacy_models', 'models': {
            'gmsc_probit': {'redevelop': {'data': str(data), 'reference': str(reference),
                                          'tier': 'reestimation', 'baseline': 'gmsc_probit'},
                            'validation_policy': str(base / 'policy.json')},
            'gmsc_lightgbm': {'redevelop': None}}}), encoding='utf-8')
        for name, value in [('CONFIG', config), ('ARTIFACTS', base / 'artifacts')]:
            patcher = patch.object(pipeline, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def approve(self, group, resolution=''):
        pipeline.review(group, 'Reviewer', 'approve', 'Checked against the source', resolution)

    def test_full_offline_run_pauses_at_every_gate(self):
        quiet = dict(progress=lambda message: None)
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', 'gate1'))
        self.approve('gmsc_probit')
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', 'gate2'))
        self.approve('gmsc_probit', 'Offline draft checked by hand')
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', 'gate3'))
        self.approve('gmsc_probit')
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', 'gate4'))
        state = pipeline.load_state('gmsc_probit')
        validation = json.loads(Path(state['steps']['validate']['state_path']).read_text(encoding='utf-8'))
        self.assertTrue(Path(validation['docx_path']).is_file())
        self.assertEqual(pipeline.load_registry()['models'][state['model_id']]['stage'], 'Staging')
        self.approve('gmsc_probit', 'Unknowns accepted for research use')
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('done', 'validated_for_research'))
        entry = pipeline.load_registry()['models'][state['model_id']]
        self.assertEqual([h['stage'] for h in entry['history']], ['Staging', 'Validated (research)'])
        self.render_app()

    def render_app(self):
        """Smoke test: both UI views render against a finished pipeline without exceptions."""
        from streamlit.testing.v1 import AppTest
        script = str(Path(__file__).parents[1] / 'src' / 'ui' / 'app.py')
        dashboard = AppTest.from_file(script, default_timeout=120).run()
        self.assertFalse(dashboard.exception, dashboard.exception)
        self.assertIn('Model risk control ledger', [t.value for t in dashboard.title])
        html = ' '.join(m.value for m in dashboard.markdown)
        self.assertIn("<b>2</b><span>models in scope</span>", html)
        self.assertIn("Overall rating", html)
        console = AppTest.from_file(script, default_timeout=120).run()
        console.sidebar.radio[0].set_value('Gate console').run()
        self.assertFalse(console.exception, console.exception)
        self.assertIn('Gate console', [t.value for t in console.title])
        gates = [e['gate'] for e in pipeline.load_state('gmsc_probit')['history'] if e['event'] == 'gate_decision']
        self.assertEqual(gates, ['gate1', 'gate2', 'gate3', 'gate4'])

    def test_rejection_reruns_the_stage_and_clears_later_steps(self):
        quiet = dict(progress=lambda message: None)
        pipeline.run('gmsc_lightgbm', False, **quiet)
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            pipeline.review('gmsc_lightgbm', 'Reviewer', 'approve', 'Checked', corrections={'colour': 'x'})
        pipeline.review('gmsc_lightgbm', 'Reviewer', 'approve', 'Checked',
                        corrections={'output': 'Default probability per applicant'})
        gate1 = json.loads(Path(pipeline.load_state('gmsc_lightgbm')['steps']['gate1']['review_path'])
                           .read_text(encoding='utf-8'))
        self.assertEqual(gate1['approval']['corrections']['output']['corrected'], 'Default probability per applicant')
        pipeline.run('gmsc_lightgbm', False, **quiet)
        first = pipeline.load_state('gmsc_lightgbm')['steps']['document']['state_path']
        pipeline.review('gmsc_lightgbm', 'Reviewer', 'reject', 'Inputs list is wrong')
        self.assertEqual(pipeline.next_action(pipeline.load_state('gmsc_lightgbm')), ('run', 'document'))
        self.assertEqual(pipeline.run('gmsc_lightgbm', False, **quiet), ('await', 'gate2'))
        state = pipeline.load_state('gmsc_lightgbm')
        self.assertNotEqual(state['steps']['document']['state_path'], first)
        self.assertNotIn('gate2', state['steps'])
        self.approve('gmsc_lightgbm', 'Offline draft checked by hand')
        self.assertEqual(pipeline.run('gmsc_lightgbm', False, **quiet), ('done', 'documented'))
        with self.assertRaisesRegex(ValueError, 'No gate is waiting'):
            self.approve('gmsc_lightgbm')

    def test_review2_can_return_to_review1(self):
        quiet = dict(progress=lambda message: None)
        for _ in range(2):
            pipeline.run('gmsc_probit', False, **quiet)
            self.approve('gmsc_probit', 'Offline draft checked by hand')
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', 'gate3'))
        with self.assertRaisesRegex(ValueError, 'only a gate 3 or gate 4 rejection'):
            pipeline.review('gmsc_probit', 'Reviewer', 'approve', 'Fine', return_to_review1=True, issue='x')
        with self.assertRaisesRegex(ValueError, 'needs the issue'):
            pipeline.review('gmsc_probit', 'Reviewer', 'reject', 'Rows differ', return_to_review1=True)
        first = pipeline.load_state('gmsc_probit')['steps']['redevelop']['state_path']
        pipeline.review('gmsc_probit', 'Reviewer', 'reject', 'Rows differ', return_to_review1=True,
                        issue='Past-due filter in the document')
        state = pipeline.load_state('gmsc_probit')
        self.assertEqual(pipeline.next_action(state), ('await', 'gate2'))
        self.assertEqual(state['review1_issue']['issue'], 'Past-due filter in the document')
        self.assertEqual(state['history'][-1]['event'], 'returned_to_review1')
        from streamlit.testing.v1 import AppTest
        console = AppTest.from_file(str(Path(__file__).parents[1] / 'src' / 'ui' / 'app.py'), default_timeout=120).run()
        console.sidebar.radio[0].set_value('Gate console').run()
        self.assertTrue(any('Past-due filter' in w.value for w in console.warning), console.exception)
        self.approve('gmsc_probit', 'Filter wording fixed in the draft')
        self.assertNotIn('review1_issue', pipeline.load_state('gmsc_probit'))
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', 'gate3'))
        self.assertNotEqual(pipeline.load_state('gmsc_probit')['steps']['redevelop']['state_path'], first)

    def test_latest_approval_can_be_revoked(self):
        quiet = dict(progress=lambda message: None)
        for _ in range(2):
            pipeline.run('gmsc_probit', False, **quiet)
            self.approve('gmsc_probit', 'Offline draft checked by hand')
        pipeline.run('gmsc_probit', False, **quiet)
        self.approve('gmsc_probit')
        with self.assertRaisesRegex(ValueError, 'needs a reviewer and a reason'):
            pipeline.revoke('gmsc_probit', 'Redevelop reviewer', ' ')
        self.assertEqual(pipeline.revoke('gmsc_probit', 'Redevelop reviewer', 'IR row id is the target'), 'gate3')
        state = pipeline.load_state('gmsc_probit')
        self.assertEqual(pipeline.next_action(state), ('await', 'gate3'))
        self.assertEqual(state['history'][-1]['event'], 'approval_revoked')
        self.assertEqual(pipeline.load_registry()['models'][state['model_id']]['stage'], 'Revoked')
        pipeline.review('gmsc_probit', 'Redevelop reviewer', 'reject', 'Rebuild with a correct IR')
        self.assertEqual(pipeline.next_action(pipeline.load_state('gmsc_probit')), ('run', 'redevelop'))
        with self.assertRaisesRegex(ValueError, 'must be an approval'):
            pipeline.revoke('gmsc_probit', 'Redevelop reviewer', 'again')

    def test_stage_failure_is_reported_not_skipped(self):
        quiet = dict(progress=lambda message: None)
        with patch.object(pipeline, 'run_discovery', side_effect=ValueError('scan broke')):
            self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('error', 'discover'))
        state = pipeline.load_state('gmsc_probit')
        self.assertIn('scan broke', state['last_error']['error'])
        self.assertEqual(pipeline.next_action(state), ('run', 'discover'))

    def test_env_loader_keeps_existing_values(self):
        env = Path(tempfile.mkdtemp()) / '.env'
        env.write_text('PIPELINE_TEST_A=from_file\nPIPELINE_TEST_B=from_file\n# comment\n', encoding='utf-8')
        with patch.dict('os.environ', {'PIPELINE_TEST_A': 'existing'}):
            pipeline.load_env(env)
            import os
            self.assertEqual((os.environ['PIPELINE_TEST_A'], os.environ['PIPELINE_TEST_B']), ('existing', 'from_file'))


if __name__ == '__main__':
    unittest.main()
