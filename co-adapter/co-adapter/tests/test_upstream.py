"""Optional CPU contracts against a separately obtained official checkout.

Set CO_UPSTREAM_ROOT to its MSE-ChatGLM3-6B directory. No assets are loaded.
"""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import warnings

from fixtures import Config
import torch
from torch import nn

from co_adapter.adapter import install_adapter
from co_adapter.engine import no_cache
from co_adapter.upstream import DATASETS, OfficialRuntime, configure


@unittest.skipUnless(os.environ.get('CO_UPSTREAM_ROOT'), 'External official checkout not requested')
class OfficialContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = OfficialRuntime(os.environ['CO_UPSTREAM_ROOT'])
        from models.subNets.Textmodel import Language_model
        from models.ChatGLM3.modeling_chatglm import ChatGLMForConditionalGeneration
        from models.ChatGLM3.configuration_chatglm import ChatGLMConfig
        cls.language_cls = Language_model
        cls.chatglm_cls = ChatGLMForConditionalGeneration
        cls.chatglm_config = ChatGLMConfig

    def test_runtime_configuration_all_four_datasets(self):
        for dataset, (relative, mode, metric, patience, budget) in DATASETS.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = configure(dataset, 2222, root / 'weights', root / 'data', 'cpu')
                self.assertEqual(args.dataPath, str(root / 'data' / relative))
                self.assertEqual(args.pretrain_LM, str(root / 'weights'))
                self.assertEqual(args.train_mode, mode)
                self.assertEqual((args.KeyEval, args.early_stop, args.max_new_tokens), (metric, patience, budget))
                self.assertEqual((args.batch_size, args.learning_rate, args.update_epochs), (8, 5e-5, 1))
                self.assertEqual(args.device, torch.device('cpu'))

    def test_original_answer_labels_keep_padding_and_do_not_add_eos(self):
        language = self.language_cls.__new__(self.language_cls)
        nn.Module.__init__(language)
        language.device = torch.device('cpu')
        received = []

        class Tokenizer:
            eos_token_id = 99

            def __call__(self, text, **kwargs):
                received.append((text, kwargs))
                return {'input_ids': torch.tensor([[4, 5], [6, 0]])}

        language.tokenizer = Tokenizer()
        prefix = torch.zeros(2, 4, dtype=torch.long)
        for mode, labels, expected in (('regression', torch.tensor([-.54, 1.04]), ['-0.5', '1.0']),
                                       ('classification', torch.tensor([0, 6]), ['0', '6'])):
            language.train_mode = mode
            tokens, targets = language.input_labels_construct(prefix, labels, 'train')
            self.assertEqual(received[-1][0], expected)
            self.assertFalse(received[-1][1]['add_special_tokens'])
            self.assertEqual(tokens.shape[1], 6)
            self.assertTrue(targets[:, :4].eq(-100).all())
            self.assertEqual(targets[1, -1].item(), 0)
            self.assertFalse(targets.eq(99).any())

    def test_original_generation_parser_and_settings_unchanged(self):
        language = self.language_cls.__new__(self.language_cls)
        nn.Module.__init__(language)
        language.multimodal_prompt_wrap = lambda fusion: fusion
        language.input_processing = lambda fusion, mode: (torch.zeros(3, 4, dtype=torch.long), None)
        calls = []

        class Backbone(nn.Module):
            def stream_generate(self, inputs, **kwargs):
                calls.append(kwargs)
                yield torch.cat((inputs, torch.ones(3, 2, dtype=torch.long)), 1)

        class Tokenizer:
            def batch_decode(self, outputs):
                return ['：一0.2/', 'invalid', '9']

        language.model, language.tokenizer = Backbone(), Tokenizer()
        for mode, expected, budget in (('regression', [-.2, 0., 9.], 4),
                                       ('classification', [0., 0., 9.], 2)):
            language.train_mode, language.max_new_tokens = mode, budget
            model = SimpleNamespace(LLM=language)
            with no_cache(model):
                actual = language.generate(torch.zeros(3, 4, 8))
            self.assertEqual(actual, expected)
            self.assertEqual(calls[-1]['max_new_tokens'], budget)
            self.assertFalse(calls[-1]['do_sample'])
            self.assertEqual((calls[-1]['num_beams'], calls[-1]['top_k']), (1, 10))
            self.assertFalse(calls[-1]['use_cache'])
            self.assertNotIn('attention_mask', calls[-1])
            self.assertNotIn('eos_token_id', calls[-1])

    def test_actual_chatglm_full_prefix_no_cache_and_shifted_loss(self):
        cfg = self.chatglm_config(num_layers=1, padded_vocab_size=32, hidden_size=16,
            ffn_hidden_size=32, kv_channels=4, num_attention_heads=4,
            seq_length=128, eos_token_id=31, pad_token_id=0, torch_dtype=torch.float32,
            original_rope=True)
        backbone = self.chatglm_cls(cfg, empty_init=False, device='cpu').eval()
        # Some official layers intentionally skip initialization; initialize the tiny fixture.
        with torch.no_grad():
            for parameter in backbone.parameters():
                parameter.normal_(0, .02)
        inputs = torch.tensor([[1, 2, 3, 4]])
        fusion = torch.randn(1, 3, 16, requires_grad=True)
        labels = torch.tensor([[-100, -100, 7, 0]])
        output = backbone(input_ids=inputs, input_fusion=fusion, labels=labels, use_cache=False)
        expected = nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
        self.assertTrue(torch.allclose(output.loss, expected))
        output.loss.backward()
        self.assertIsNotNone(fusion.grad)
        lengths, caches = [], []

        def observe(module, args, kwargs):
            lengths.append(kwargs['input_ids'].shape[1])
            caches.append((kwargs.get('use_cache'), kwargs.get('past_key_values')))

        handle = backbone.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            with no_cache(SimpleNamespace(LLM=SimpleNamespace(model=backbone))), torch.no_grad():
                list(backbone.stream_generate(inputs, input_fusion=fusion.detach(),
                    max_new_tokens=3, min_new_tokens=3, do_sample=False))
        finally:
            handle.remove()
        self.assertEqual(lengths, [4, 5, 6])
        self.assertTrue(all(cache is False and past is None for cache, past in caches))

    def test_official_lstm_allocation_and_co_unused_projection(self):
        from models.multiTask.CMCM import TVA_LSTM
        model = nn.Module()
        model.audio_LSTM = TVA_LSTM(25, 64, dropout=0)
        model.video_LSTM = TVA_LSTM(177, 64, dropout=0)
        model.text_guide_mixer = nn.Identity()
        model.mutli_scale_fusion = nn.Identity()
        install_adapter(model)
        allocated = sum(p.numel() for p in model.parameters())
        unused = sum(p.numel() for p in model.independent.text_query.parameters())
        self.assertEqual(allocated, 1866624)
        self.assertEqual(allocated - unused, 1342208)


if __name__ == '__main__':
    unittest.main()
