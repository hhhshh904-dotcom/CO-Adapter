import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fixtures import Config, Runtime, TinyEncoder, config
import numpy as np
import torch

from co_adapter import engine
from co_adapter.adapter import IndependentTokens, temporal_memory
from co_adapter.state import (model_state, read_json, restore_model, rng_state,
                              restore_rng, seed_all)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        seed_all(7, torch.device('cpu'))
        self.text = torch.randn(2, 3, 8)
        self.mask = torch.tensor([[True, True, False], [True, False, False]])
        self.memories = [(torch.randn(2, 4, 256), torch.tensor([[True, True, False, False],
                                                             [True, False, False, False]])) for _ in range(2)]

    def test_co_is_text_independent_and_has_four_tokens(self):
        adapter = IndependentTokens(8, 8)
        first = adapter(self.text, self.mask, self.memories)
        second = adapter(self.text + 30, ~self.mask, self.memories)
        self.assertEqual(first.shape, (2, 4, 8))
        self.assertTrue(torch.equal(first, second))
        first.sum().backward()
        self.assertIsNone(adapter.text_query.weight.grad)
        self.assertIsNotNone(adapter.queries.grad)

    def test_mixed_conditions_only_second_token(self):
        adapter = IndependentTokens(8, 8, 'mixed')
        first = adapter(self.text, self.mask, self.memories)
        second = adapter(self.text + torch.arange(8), self.mask, self.memories)
        self.assertTrue(torch.equal(first[:, [0, 2]], second[:, [0, 2]]))
        self.assertFalse(torch.equal(first[:, [1, 3]], second[:, [1, 3]]))
        first.sum().backward()
        self.assertIsNotNone(adapter.text_query.weight.grad)

    def test_masked_padding_is_not_read_and_empty_modality_is_zero(self):
        adapter = IndependentTokens(8, 8)
        first = adapter(self.text, self.mask, self.memories)
        changed = [(m.masked_fill(~mask[..., None], 10000), mask) for m, mask in self.memories]
        self.assertTrue(torch.equal(first, adapter(self.text, self.mask, changed)))
        empty = [(m, torch.zeros_like(mask)) for m, mask in self.memories]
        self.assertEqual(adapter(self.text, self.mask, empty).count_nonzero().item(), 0)

    def test_temporal_memory_lengths_and_all_valid_states(self):
        encoder = TinyEncoder().eval()
        values = torch.randn(2, 4, 3)
        memory, mask = temporal_memory(encoder, values, torch.tensor([4, 1]))
        self.assertEqual(memory.shape, (2, 4, 256))
        self.assertEqual(mask.sum().item(), 5)
        self.assertEqual(memory[1, 1:].count_nonzero().item(), 0)
        changed = values.clone()
        changed[1, 1:] = 10000
        self.assertTrue(torch.equal(memory, temporal_memory(encoder, changed, torch.tensor([4, 1]))[0]))
        for lengths in ([0, 1], [5, 1]):
            with self.assertRaises(ValueError):
                temporal_memory(encoder, values, torch.tensor(lengths))

    def test_allocation_kept_for_both_variants(self):
        co, mixed = IndependentTokens(4096), IndependentTokens(4096, variant='mixed')
        self.assertEqual(sum(p.numel() for p in co.parameters()), sum(p.numel() for p in mixed.parameters()))
        self.assertEqual(co.text_query.weight.numel() + co.text_query.bias.numel(), 524416)


class EngineTests(unittest.TestCase):
    def test_resource_change_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            engine.train(config(), 'co', out, Runtime(), {'source': 'before'}, smoke=True)
            with self.assertRaisesRegex(ValueError, 'changed'):
                engine.train(config(), 'co', out, Runtime(), {'source': 'after'}, resume=True, smoke=True)

    def test_current_source_inventory_changes(self):
        from co_adapter.provenance import inventory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models').mkdir()
            source = root / 'models' / 'small.py'
            source.write_text('value = 1\n')
            before = inventory(root)
            source.write_text('value = 2\n')
            after = inventory(root)
            self.assertNotEqual(before['upstream_sources'], after['upstream_sources'])
            self.assertIn('src/co_adapter/engine.py', before['release_sources'])
            self.assertIn('scipy', before['dependencies'])

    def test_rng_roundtrip(self):
        import random
        saved = rng_state()
        before = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(saved)
        after = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(before[:2], after[:2])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_selection_and_parser_diagnostics(self):
        self.assertTrue(engine.improves(1, None, 'MAE'))
        self.assertFalse(engine.improves(1, 1, 'MAE'))
        self.assertTrue(engine.improves(.9, 1, 'MAE'))
        self.assertTrue(engine.improves(.6, .5, 'weight_F1'))
        with self.assertRaises(FloatingPointError):
            engine.improves(None, .5, 'MAE')
        self.assertTrue(engine.parses('：一0.2/', True))
        self.assertFalse(engine.parses('invalid', True))
        self.assertFalse(engine.parses('nan', False))

    def test_no_cache_restores_even_on_error(self):
        model = Runtime().make_model(config(), 'co')
        original = model.LLM.model.stream_generate
        with self.assertRaises(RuntimeError):
            with engine.no_cache(model):
                list(model.LLM.model.stream_generate(torch.zeros(1), use_cache=True))
                raise RuntimeError('probe')
        self.assertEqual(model.LLM.model.stream_generate, original)
        self.assertFalse(model.LLM.model.calls[0]['use_cache'])

    def test_strict_restore(self):
        model = Runtime().make_model(config(), 'co')
        saved = model_state(model)
        bad = copy.deepcopy(saved)
        bad['parameters'].pop(next(iter(bad['parameters'])))
        with self.assertRaises(ValueError):
            restore_model(model, bad)

    def test_interrupt_resume_is_exact(self):
        for variant in ('co', 'mixed'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                a, b = Path(directory) / 'a', Path(directory) / 'b'
                engine.train(config(), variant, a, Runtime(), {}, smoke=True)
                with self.assertRaises(InterruptedError):
                    engine.train(config(), variant, b, Runtime(), {}, smoke=True, stop_after_batches=3)
                state = torch.load(b / 'latest.pt', weights_only=False)
                self.assertEqual(state['state']['batch'], 3)
                self.assertIn('optimizer', state)
                self.assertIn('scaler', state)
                engine.train(config(), variant, b, Runtime(), {}, resume=True, smoke=True)
                first = torch.load(a / 'latest.pt', weights_only=False)
                second = torch.load(b / 'latest.pt', weights_only=False)
                for name in first['parameters']:
                    self.assertTrue(torch.equal(first['parameters'][name], second['parameters'][name]), name)
                self.assertEqual(first['state'], second['state'])
                self.assertTrue(torch.equal(first['rng']['torch'], second['rng']['torch']))
                self.assertEqual(read_json(a / 'valid_001.json'), read_json(b / 'valid_001.json'))
                for group in second['optimizer']['param_groups']:
                    self.assertEqual(group['lr'], 5e-5)
                    self.assertEqual(group['eps'], 1e-4)
                    self.assertEqual(group['weight_decay'], .01)

    def test_same_process_reload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            runtime = Runtime()
            engine.train(config(), 'co', out, runtime, {}, smoke=True)
            with self.assertRaisesRegex(RuntimeError, 'fresh process'):
                engine.replay_and_evaluate(config(), 'co', out, runtime, {}, smoke=True)
            self.assertNotIn('test', runtime.loaded)

    def test_no_training_before_smoke_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime()
            with self.assertRaises(FileNotFoundError):
                engine.train(config(), 'co', Path(directory), runtime, {})
            self.assertEqual(runtime.loaded, [])


class FreshProcessTests(unittest.TestCase):
    def run_stage(self, command, out, variant='co', mode='regression', success=True):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tests' / 'fixtures.py'), command,
                                 str(out), variant, mode], env=env, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_fresh_smoke_train_test_both_variants_and_task_modes(self):
        for variant, mode in (('co', 'regression'), ('mixed', 'classification')):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                self.run_stage('smoke', out, variant, mode)
                self.run_stage('reload-smoke', out, variant, mode)
                self.assertNotIn('test', (out / 'access.txt').read_text().splitlines())
                self.run_stage('train', out, variant, mode)
                best = torch.load(out / 'best.pt', weights_only=False)
                self.assertEqual(best['epoch'], 1)
                latest = torch.load(out / 'latest.pt', weights_only=False)
                self.assertEqual(latest['state']['epoch'], 3)
                self.assertNotIn('test', (out / 'access.txt').read_text().splitlines())
                # Corrupt complete validation JSON: no test access allowed.
                valid_path = out / 'valid_001.json'
                original = valid_path.read_text()
                changed = json.loads(original)
                changed['predictions'][0]['raw'] = 'changed'
                valid_path.write_text(json.dumps(changed))
                failed = self.run_stage('evaluate', out, variant, mode, success=False)
                self.assertIn('held-out test forbidden', failed.stderr)
                self.assertNotIn('test', (out / 'access.txt').read_text().splitlines())
                valid_path.write_text(original)
                self.run_stage('evaluate', out, variant, mode)
                self.assertEqual((out / 'access.txt').read_text().splitlines().count('test'), 1)
                self.assertEqual(read_json(out / 'reload.json'), read_json(valid_path))
                self.run_stage('evaluate', out, variant, mode, success=False)
                self.assertEqual((out / 'access.txt').read_text().splitlines().count('test'), 1)

    def test_incomplete_training_cannot_test(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.run_stage('smoke', out)
            self.run_stage('reload-smoke', out)
            self.run_stage('evaluate', out, success=False)
            self.assertNotIn('test', (out / 'access.txt').read_text().splitlines())

    def test_cli_help_and_option_validation(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'train.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--upstream-root', result.stdout)
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'train.py'), 'smoke', '--smoke-only'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
