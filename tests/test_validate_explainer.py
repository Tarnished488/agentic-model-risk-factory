import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from docx import Document

from src.discover.discover_agent import atomic_json
from src.redevelop.redevelop_agent import CheckError
from src.validate.explainer import check_explanation, template_explanation, unsupported_numbers
from src.validate.report_docx import write_report
from src.validate.validate_agent import DEFAULT_POLICY, coefficient_signs, evaluate, explain, read_policy


class ExplainerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        rows = 400
        pd.DataFrame({'id': np.arange(rows).astype(str), 'target': np.tile([0, 1], rows // 2),
                      'age': np.tile([25, 25, 55, 55], rows // 4), 'util': np.tile([0., 1.], rows // 2)}
                     ).to_csv(self.root / 'raw.csv', index=False)
        pd.DataFrame({'row_id': np.arange(200)}).to_csv(self.root / 'train.csv', index=False)
        pd.DataFrame({'row_id': np.arange(200, 400), 'probability': np.tile([.05, .95], 100),
                      'label': np.tile([0, 1], 100)}).to_csv(self.root / 'pred.csv', index=False)
        pd.DataFrame({'term': ['(Intercept)', 'unsecured.credit', 'age'], 'Estimate': [-1.8, 0.9, 0.01]}
                     ).to_csv(self.root / 'coef.csv', index=False)
        self.ir = {'data': {'row_id': 'id'}, 'filters': [],
                   'rename': {'mapping': {'util': 'unsecured.credit'}},
                   'model': {'target': 'target', 'features': ['util', 'age']}}
        atomic_json(self.root / 'ir.json', self.ir)
        self.state = {'model_id': 'TEST', 'ir_path': str(self.root / 'ir.json'), 'data_path': str(self.root / 'raw.csv'),
                      'train_ids_path': str(self.root / 'train.csv'), 'coefficients_path': str(self.root / 'coef.csv'),
                      'holdout_predictions_path': str(self.root / 'pred.csv'), 'approvals': []}
        self.policy = {**read_policy(DEFAULT_POLICY), 'expected_signs': {'util': '+', 'age': '-'}}

    def report(self):
        return {'model_id': 'TEST', 'generated_at': 'now', 'policy': self.policy, **evaluate(self.state, self.policy)}

    def test_number_check_accepts_rounding_and_rejects_invention(self):
        allowed = [0.8230755, 0.057577, 33607, -0.0072249]
        text = ('AUC 0.823 (0.82), 5.8% bad rate on 33,607 rows of MODEL-9239165C4AE5; '
                'NumberOfTime30-59DaysPastDue; age coefficient -0.0072; two groups and 3 tiers.')
        self.assertEqual(unsupported_numbers(text, allowed), [])
        self.assertEqual(unsupported_numbers('AUC 0.83 under GDPR Article 22, gap 12.5%', allowed),
                         ['0.83', '22', '12.5%'])

    def test_coefficient_signs_use_the_rename_mapping(self):
        coefficients, findings = coefficient_signs(self.state, self.ir, self.policy)
        self.assertEqual(coefficients, {'util': 0.9, 'age': 0.01})
        self.assertEqual({f['check']: f['result'] for f in findings},
                         {'coefficient_sign:util': 'pass', 'coefficient_sign:age': 'fail'})
        self.assertEqual(coefficient_signs({**self.state, 'coefficients_path': None}, self.ir, self.policy)[1][0]['result'],
                         'unverifiable')

    def test_conclusion_must_follow_the_rules(self):
        report = self.report()
        self.assertEqual(report['status'], 'needs_attention')  # the age sign fails
        explanation = template_explanation(report)
        self.assertEqual(explanation.conclusion, 'fail')
        with self.assertRaisesRegex(CheckError, 'Conclusion'):
            check_explanation(explanation.model_copy(update={'conclusion': 'pass'}), report)

    def test_llm_explainer_retries_invented_numbers_then_falls_back(self):
        report = self.report()
        good = template_explanation(report).model_dump()
        invented = {**good, 'summary': 'AUC is 0.91 and the model is approved under Article 22.'}
        answers = iter([json.dumps(invented), json.dumps(good)])
        with patch('src.redevelop.redevelop_agent.time.sleep'):
            explanation, mode, error = explain(report, self.root, True, lambda m, s: next(answers))
        self.assertEqual((mode, error), ('llm', None))
        audit = (self.root / 'audit_log.jsonl').read_text(encoding='utf-8')
        self.assertIn("Numbers not found in the metrics payload: ['0.91', '22']", audit)
        with patch('src.redevelop.redevelop_agent.time.sleep'):
            explanation, mode, error = explain(report, self.root, True, lambda m, s: json.dumps(invented))
        self.assertEqual(mode, 'template_after_llm_failure')
        self.assertIn('0.91', error)

    def test_rag_ratings_and_recommendations(self):
        from src.validate.ratings import annotate, rate
        amber = self.policy['amber']
        self.assertEqual(rate({'check': 'holdout_auc', 'result': 'pass', 'value': 0.68}, amber), 'amber')
        self.assertEqual(rate({'check': 'holdout_auc', 'result': 'pass', 'value': 0.82}, amber), 'green')
        self.assertEqual(rate({'check': 'age_disparate_impact', 'result': 'pass', 'value': 0.85}, amber), 'amber')
        self.assertEqual(rate({'check': 'train_holdout_psi:x', 'result': 'pass', 'value': 0.2}, amber), 'amber')
        self.assertEqual(rate({'check': 'anything', 'result': 'fail', 'value': 1}, amber), 'red')
        self.assertEqual(rate({'check': 'anything', 'result': 'unverifiable'}, amber), 'amber')
        findings = [{'check': 'coefficient_sign:age', 'result': 'fail', 'value': 0.1}]
        self.assertEqual(annotate(findings, self.policy), 'red')
        self.assertIn('direction', findings[0]['recommendation'])
        inverted = [{'check': 'score_direction', 'result': 'fail', 'value': 0.27, 'limit': 0.5}]
        self.assertEqual(annotate(inverted, self.policy), 'red')
        self.assertIn('wrong class', inverted[0]['recommendation'])

    def test_fairlearn_metrics_use_approval_for_disparate_impact(self):
        from src.validate.fairness import fairness_audit
        rows = 600
        raw = pd.DataFrame({'sex': np.repeat(['male', 'female'], rows // 2), 'age': np.tile([30, 50], rows // 2)},
                           index=[str(i) for i in range(rows)])
        y = np.tile([0, 0, 1], rows // 3)
        scores = np.where(np.arange(rows) < rows // 2, 0.9 * y + 0.05, 0.5 * y + 0.1)  # women flagged less
        policy = {**self.policy, 'fairness_attributes': [{'column': 'sex'}, {'column': 'ethnicity'}]}
        groups, summary, findings, _ = fairness_audit(raw, raw.index, y, scores, policy)
        flag_rates = {g['group']: g['selection_rate'] for g in groups}
        approval = {g['group']: g['approval_rate'] for g in groups}
        self.assertAlmostEqual(summary['sex']['disparate_impact'], min(approval.values()) / max(approval.values()))
        self.assertAlmostEqual(summary['sex']['demographic_parity_difference'],
                               abs(flag_rates['male'] - flag_rates['female']))
        results = {f['check']: f['result'] for f in findings}
        self.assertEqual(results['ethnicity_group_fairness'], 'unverifiable')
        self.assertIn('sex_equalized_odds_difference', results)

    def test_charts_and_findings_file_are_written(self):
        from src.validate.charts import draw_charts
        from src.validate.ratings import annotate
        report = self.report()
        report['overall_rating'] = annotate(report['findings'], self.policy)
        charts = draw_charts(report, self.root / 'charts')
        self.assertEqual(set(charts), {'roc', 'calibration', 'score_distribution', 'fairness', 'psi'})
        self.assertTrue(all(Path(p).stat().st_size > 1000 for p in charts.values()))
        self.assertEqual(report['curves']['roc'][0], {'fpr': 0.0, 'tpr': 0.0})
        self.assertAlmostEqual(report['curves']['roc'][-1]['tpr'], 1.0)

    def test_word_report_contains_metrics_explanation_and_governance(self):
        report = self.report()
        explanation = template_explanation(report).model_dump()
        report.update(explanation=explanation, explainer_mode='template')
        approvals = [{'gate': 'redevelop', 'reviewer': 'Redevelop reviewer', 'decision': 'approve', 'time': 't'}]
        path = write_report(report, explanation, approvals, self.root / 'report.docx')
        text = '\n'.join(p.text for p in Document(path).paragraphs)
        cells = {c.text for t in Document(path).tables for r in t.rows for c in r.cells}
        self.assertIn('Independent Model Validation Report', text)
        self.assertIn(explanation['summary'], text)
        self.assertIn('coefficient_sign:age', cells)
        self.assertIn('Redevelop reviewer', cells)


if __name__ == '__main__':
    unittest.main()
