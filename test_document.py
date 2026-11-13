import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from discover_agent import run_discovery
from discover_review import review
from document_agent import run_document
from document_review import review_document, load_redevelop_input
from document_sources import chunk_sources
from discover_agent import Record, LLMResponse


class DocumentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / 'models'
        self.root.mkdir()
        self.code = self.root / 'model.R'
        self.code.write_text('library(lightgbm)\n')
        self.discover = self.base / 'discover'
        model = run_discovery(self.root, self.discover).models[0]
        self.approved = review(self.discover, model.model_id, 'Test reviewer', 'approve', 'Test fixture')
        self.output = self.base / 'document'
        self.sleep = patch("document_agent.time.sleep")
        self.sleep_mock = self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def run_agent(self, **kwargs):
        path = run_document(self.approved, self.output, **kwargs)
        state = json.loads(path.read_text())
        return state, json.loads(Path(state['consistency_report_path']).read_text())

    def test_offline_draft_and_lineage(self):
        state, report = self.run_agent()
        self.assertEqual(state['status'], 'pending_document_review')
        self.assertEqual(report['mode'], 'offline')
        self.assertTrue(all(c['result'] == 'unverifiable' for c in report['items']))
        self.assertEqual(report['record_sha256'], state['approvals'][0]['record_sha256'])
        self.assertIn('Model Purpose and Scope', Path(state['doc_path']).read_text())

    def responder(self, verdict='consistent'):
        evidence = {'source_id': 'model.R', 'line_start': 1, 'line_end': 1, 'quote': 'library(lightgbm)'}
        def call(messages, schema):
            if schema['title'] == 'Section':
                return json.dumps({'claims': [{'statement': 'Loads the lightgbm package', 'evidence': [evidence]}],
                                   'unknowns': ['Business use unknown'], 'recommendations': []})
            payload = json.loads(messages[1]['content'])
            return json.dumps({'items': [{'claim_id': cid, 'result': verdict, 'reason': 'Checked against source code',
                                          'evidence': [evidence]} for cid in payload['claims']]})
        return call

    def test_generation_and_complete_self_check(self):
        state, report = self.run_agent(use_llm=True, caller=self.responder())
        self.assertEqual(state['mode'], 'llm')
        self.assertEqual(len(report['items']), 6)
        self.assertEqual(state['status'], 'pending_document_review')

    def test_contradiction_needs_attention(self):
        state, _ = self.run_agent(use_llm=True, caller=self.responder('inconsistent'))
        self.assertEqual(state['status'], 'needs_attention')

    def test_invalid_json_retries_and_explicit_failure(self):
        calls = []
        def invalid(messages, schema):
            calls.append(1)
            return '{}'
        state, report = self.run_agent(use_llm=True, caller=invalid)
        self.assertEqual(len(calls), 3)
        self.assertEqual(state['status'], 'needs_attention')
        self.assertEqual(state['mode'], 'partial_llm')
        self.assertIsNotNone(report['failure'])

    def test_missing_self_check_claims_fail(self):
        good = self.responder()
        def partial(messages, schema):
            return '{"items": []}' if schema['title'] == 'Checks' else good(messages, schema)
        state, _ = self.run_agent(use_llm=True, caller=partial)
        self.assertEqual(state['status'], 'needs_attention')

    def test_forged_quote_fails(self):
        good = self.responder()
        state, _ = self.run_agent(use_llm=True, caller=lambda m, s: good(m, s).replace('library(lightgbm)', 'fake()'))
        self.assertEqual(state['status'], 'needs_attention')

    def test_source_mutation_during_generation_blocks_output(self):
        good = self.responder()
        def mutate(messages, schema):
            self.code.write_text('changed')
            return good(messages, schema)
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.run_agent(use_llm=True, caller=mutate)
        self.assertFalse(list(self.output.rglob('state.json')))

    def test_rejected_input_and_output_boundary(self):
        payload = json.loads(self.approved.read_text())
        payload['status'] = 'rejected'
        rejected = self.base / 'rejected.json'
        rejected.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'approval'):
            run_document(rejected, self.output)
        with self.assertRaisesRegex(ValueError, 'outside'):
            run_document(self.approved, self.root / 'docs')

    def test_stage_windows_are_lossless_and_keep_original_lines(self):
        self.code.write_text('x <- read.csv("x.csv")\nx$age <- ifelse(is.na(x$age), 0, x$age)\n'
                             'm <- glm(y ~ age, data=x)\np <- predict(m, x)\n')
        inv = run_discovery(self.root, self.discover)
        chunks = chunk_sources(inv.models[0], 256)
        self.assertEqual([c['stage'] for c in chunks], ['data', 'variables', 'fit', 'scoring'])
        lines = [line for c in chunks for line in c['lines']]
        self.assertEqual(lines, [f'{i}: {line}' for i, line in enumerate(self.code.read_text().splitlines(), 1)])
        self.assertEqual(chunks[-1]['line_start'], 4)

    def test_long_sources_split_without_truncation(self):
        record = Record.model_validate(json.loads(self.approved.read_text())['inventory_record'])
        record.sources[0].text = '\n'.join(f'# line {i} ' + 'x' * 30 for i in range(100))
        chunks = chunk_sources(record, 256)
        self.assertGreater(len(chunks), 10)
        self.assertEqual(sum(len(c['lines']) for c in chunks), 100)
        self.assertTrue(all(sum(len(line) + 1 for line in c['lines']) <= 256 for c in chunks))

    def test_retry_feedback_backoff_and_partial_sections(self):
        good = self.responder()
        attempts = []
        def fail_later(messages, schema):
            attempts.append([dict(m) for m in messages])
            if len(attempts) == 1:
                return good(messages, schema)
            return '{}'
        state, report = self.run_agent(use_llm=True, caller=fail_later)
        self.assertEqual(state['mode'], 'partial_llm')
        self.assertEqual(len(report['items']), 1)
        self.assertEqual(report['items'][0]['statement'], 'Loads the lightgbm package')
        self.assertIn('claims', attempts[-1][-1]['content'])
        self.assertEqual([call.args[0] for call in self.sleep_mock.call_args_list], [0.2, 0.4])
        self.assertEqual(state['retry_count'], 2)

    def test_cache_reuses_only_validated_same_inputs(self):
        calls = []
        good = self.responder()
        def caller(messages, schema):
            calls.append(1)
            return LLMResponse(good(messages, schema), {'total_tokens': 17})
        first, _ = self.run_agent(use_llm=True, caller=caller)
        self.assertEqual(len(calls), 7)
        second, _ = self.run_agent(use_llm=True, caller=caller)
        self.assertEqual(len(calls), 7)
        self.assertNotEqual(first['document_run_id'], second['document_run_id'])
        events = [json.loads(line) for line in (self.output / 'audit_log.jsonl').read_text().splitlines()]
        success = next(e for e in events if e.get('outcome') == 'success')
        self.assertEqual(success['tokens']['total_tokens'], 17)
        self.assertIn('input_hash', success)
        self.assertTrue(any(e.get('outcome') == 'cache_hit' for e in events))
        self.assertIn('ir_path', second)
        self.assertIsNone(second['ir_path'])

    def test_human_approval_freezes_edited_document(self):
        state, _ = self.run_agent(use_llm=True, caller=self.responder())
        state_path = Path(state['doc_path']).parents[1] / 'state.json'
        Path(state['doc_path']).write_text('Model description after human review')
        with self.assertRaisesRegex(ValueError, 'resolution'):
            review_document(state_path, 'Reviewer', 'approve', 'Checked')
        approved = review_document(state_path, 'Reviewer', 'approve', 'Checked', 'Checked each edit against the source code')
        result = load_redevelop_input(approved)
        self.assertEqual(result['status'], 'ready_for_redevelop')
        self.assertEqual(len(result['approvals']), 2)
        self.assertEqual(result['document_text'], 'Model description after human review')
        Path(state['doc_path']).write_text('draft edited again')
        self.assertEqual(load_redevelop_input(approved)['document_text'], 'Model description after human review')
        Path(result['doc_path']).write_text('tampered snapshot')
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_redevelop_input(approved)

    def test_reject_and_failed_run_cannot_handoff(self):
        state, _ = self.run_agent(use_llm=True, caller=self.responder())
        state_path = Path(state['doc_path']).parents[1] / 'state.json'
        rejected = review_document(state_path, 'Reviewer', 'reject', 'Correct missing-value handling')
        self.assertEqual(json.loads(rejected.read_text())['rejection_feedback'], 'Correct missing-value handling')
        with self.assertRaisesRegex(ValueError, 'approval'):
            load_redevelop_input(rejected)
        failed, _ = self.run_agent(use_llm=True, caller=lambda m, s: '{}', use_cache=False)
        with self.assertRaisesRegex(ValueError, 'rerun'):
            review_document(Path(failed['doc_path']).parents[1] / 'state.json',
                            'Reviewer', 'approve', 'Checked', 'Resolved')

    def test_report_tamper_and_source_changes_block_approval(self):
        state, _ = self.run_agent(use_llm=True, caller=self.responder())
        state_path = Path(state['doc_path']).parents[1] / 'state.json'
        report_path = Path(state['consistency_report_path'])
        original = report_path.read_text()
        report_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            review_document(state_path, 'Reviewer', 'approve', 'Checked')
        report_path.write_text(original)
        approved = review_document(state_path, 'Reviewer', 'approve', 'Checked')
        self.code.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            load_redevelop_input(approved)

    def test_mlflow_failure_keeps_local_results(self):
        with patch('document_agent.log_document_mlflow', side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(ValueError, 'local output saved'):
                self.run_agent(use_mlflow=True)
        self.assertEqual(len(list(self.output.rglob('state.json'))), 1)

    def test_rejection_feedback_routes_to_new_generation(self):
        state, _ = self.run_agent(use_llm=True, caller=self.responder())
        state_file = Path(state['doc_path']).parents[1] / 'state.json'
        rejected = review_document(state_file, 'Reviewer', 'reject', 'Clarify package dependency')
        seen = []
        good = self.responder()
        def caller(messages, schema):
            seen.append(json.loads(messages[1]['content']))
            return good(messages, schema)
        regenerated, _ = self.run_agent(use_llm=True, caller=caller, feedback_review=rejected)
        self.assertIn('Clarify package dependency', json.dumps(seen))
        self.assertEqual(regenerated['review_feedback']['approval']['decision'], 'reject')

    def test_cross_window_contradiction_is_not_lost(self):
        self.code.write_text('x <- read.csv("x.csv")\nx <- ifelse(is.na(x), 0, x)\n')
        model = run_discovery(self.root, self.discover).models[0]
        self.approved = review(self.discover, model.model_id, 'Tester', 'approve', 'Checked')
        def caller(messages, schema):
            payload = json.loads(messages[1]['content'])
            chunk = payload['sources'][0]
            line = chunk['line_start']
            ref = {'source_id': chunk['source_id'], 'line_start': line, 'line_end': line,
                   'quote': chunk['lines'][0].split(': ', 1)[1]}
            if schema['title'] == 'Section':
                return json.dumps({'claims': [{'statement': 'Missing values are imputed with the median', 'evidence': [ref]}],
                                   'unknowns': [], 'recommendations': []})
            conflicting = next(c for c in payload['sources'] if c['line_start'] == 2)
            conflict_ref = {'source_id': conflicting['source_id'], 'line_start': 2, 'line_end': 2,
                            'quote': conflicting['lines'][0].split(': ', 1)[1]}
            return json.dumps({'items': [
                {'claim_id': cid, 'result': 'inconsistent', 'reason': 'Code actually fills zero',
                 'evidence': [conflict_ref]} for cid in payload['claims']]})
        state, report = self.run_agent(use_llm=True, caller=caller)
        self.assertEqual(state['status'], 'needs_attention')
        self.assertTrue(all(i['result'] == 'inconsistent' for i in report['items']))
        self.assertTrue(all(i['evidence'][0]['line_start'] == 2 for i in report['items']))

    def test_actual_notebook_discover_to_offline_document(self):
        inventory = run_discovery(Path(__file__).parent / 'legacy_models', self.discover)
        self.approved = review(self.discover, inventory.models[0].model_id,
                               'Test fixture reviewer', 'approve', 'Integration test only')
        state, report = self.run_agent()
        self.assertEqual(state['inventory_record']['metadata']['algorithm'], 'LightGBM')
        self.assertTrue(any('cell=' in e['source_id'] for item in report['items'] for e in item['evidence']))
        chunks = chunk_sources(inventory.models[0])
        self.assertEqual(sum(len(c['lines']) for c in chunks),
                         sum(len(s.text.splitlines()) for s in inventory.models[0].sources))

    def test_cli_generate_review_handoff_round_trip(self):
        script = str(Path(__file__).parent / 'document_agent.py')
        def invoke(*arguments):
            result = subprocess.run([sys.executable, script, *arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip()
        # Original CLI form remains supported.
        state_file = invoke(str(self.approved), '--output', str(self.output))
        approved_file = invoke('review', state_file, '--reviewer', 'Test fixture',
                               '--decision', 'approve', '--notes', 'Checked',
                               '--resolution-notes', 'Manually checked offline draft against code')
        result = json.loads(invoke('handoff', approved_file))
        self.assertEqual(result['status'], 'ready_for_redevelop')

    def test_real_notebook_uses_seven_calls_instead_of_window_cross_product(self):
        inventory = run_discovery(Path(__file__).parent / 'legacy_models', self.discover)
        self.approved = review(self.discover, inventory.models[0].model_id,
                               'Test fixture', 'approve', 'Performance regression fixture')
        calls, messages = [], []
        def caller(history, schema):
            calls.append(1)
            payload = json.loads(history[1]['content'])
            chunk = payload['sources'][0]
            ref = {'source_id': chunk['source_id'], 'line_start': chunk['line_start'],
                   'line_end': chunk['line_start'], 'quote': chunk['lines'][0].split(': ', 1)[1]}
            if schema['title'] == 'Section':
                return json.dumps({'claims': [{'statement': 'Example statement', 'evidence': [ref]}],
                                   'unknowns': [], 'recommendations': []})
            return json.dumps({'items': [{'claim_id': cid, 'result': 'consistent', 'reason': 'Simulated check',
                                          'evidence': [ref]} for cid in payload['claims']]})
        state, report = self.run_agent(use_llm=True, caller=caller, progress=messages.append)
        self.assertEqual(len(calls), 7)
        self.assertEqual(report['source_groups'], 1)
        self.assertEqual(state['api_calls'], 7)
        self.assertTrue(any('Self-check:' in message for message in messages))
        self.assertGreater(len(chunk_sources(inventory.models[0])), 10)

    def test_call_budget_stops_before_extra_api_attempt(self):
        calls = []
        good = self.responder()
        def caller(messages, schema):
            calls.append(1)
            return good(messages, schema)
        state, report = self.run_agent(use_llm=True, caller=caller, max_calls=2)
        self.assertEqual(len(calls), 0)
        self.assertEqual(state['api_calls'], 0)
        self.assertEqual(state['status'], 'needs_attention')
        self.assertIn('more generation calls', report['failure'])

    def test_interrupt_preserves_sections_and_releases_lock(self):
        calls = []
        good = self.responder()
        def caller(messages, schema):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return good(messages, schema)
        state, report = self.run_agent(use_llm=True, caller=caller)
        self.assertEqual(state['status'], 'needs_attention')
        self.assertIn('Interrupted', report['failure'])
        self.assertEqual(len(report['items']), 1)
        self.assertFalse((self.output / '.discover.lock').exists())

    def test_overlong_chapter_rejected_with_bounded_retries(self):
        good = self.responder()
        def caller(messages, schema):
            payload = json.loads(good(messages, schema))
            payload['claims'] *= 13
            return json.dumps(payload)
        state, report = self.run_agent(use_llm=True, caller=caller)
        self.assertEqual(state['api_calls'], 3)
        self.assertIsNotNone(report['failure'])

    def test_retry_attempts_cannot_exceed_total_budget(self):
        calls = []
        good = self.responder()
        def caller(messages, schema):
            calls.append(1)
            return '{}' if schema['title'] == 'Checks' else good(messages, schema)
        state, report = self.run_agent(use_llm=True, caller=caller, max_calls=7)
        self.assertEqual(len(calls), 7)
        self.assertEqual(state['api_calls'], 7)
        self.assertEqual(state['status'], 'needs_attention')
        self.assertIn('budget reached', report['failure'])
        self.assertEqual(len(report['items']), 6)


if __name__ == '__main__':
    unittest.main()
