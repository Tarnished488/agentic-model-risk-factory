import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import pipeline, sandbox


class SandboxTests(unittest.TestCase):
    """The public sandbox must run every stage without real data, R or an LLM."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'sandbox'
        for target, name, value in [(sandbox, 'ROOT', self.root), (pipeline, 'CONFIG', pipeline.CONFIG),
                                    (pipeline, 'ARTIFACTS', pipeline.ARTIFACTS)]:
            patcher = patch.object(target, name, value)  # restores the real pipeline paths afterwards
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_offline_run_reaches_gate4_on_synthetic_data(self):
        sandbox.activate(pipeline)
        self.assertTrue((self.root / 'reference' / 'holdout_predictions.csv').is_file())
        self.assertTrue(str(pipeline.config()['models']['gmsc_probit']['redevelop']['data']).startswith(str(self.root)))
        quiet = dict(progress=lambda message: None)
        for gate, resolution in [('gate1', ''), ('gate2', 'Offline draft checked'), ('gate3', '')]:
            self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', gate))
            pipeline.review('gmsc_probit', 'Reviewer', 'approve', 'Checked', resolution)
        self.assertEqual(pipeline.run('gmsc_probit', False, **quiet), ('await', 'gate4'))
        sandbox.reset()
        self.assertFalse(self.root.exists())

    def test_app_shows_the_sandbox_banner(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ, {'SANDBOX': '1'}):
            app = AppTest.from_file(str(Path(__file__).parents[1] / 'src' / 'ui' / 'app.py'), default_timeout=120).run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any('synthetic rows' in w.value for w in app.warning))
        self.assertTrue((self.root / 'pipeline.json').is_file())


if __name__ == '__main__':
    unittest.main()
