"""Explicit, external official-model dependency; no vendored upstream source."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .adapter import install_adapter
from .state import rng_state, restore_rng

DATASETS = {
    'simsv2': ('SIMS_V2/ch-simsv2s.pkl', 'regression', 'MAE', 10, 4),
    'mosei': ('MOSEI/Processed/unaligned_50.pkl', 'regression', 'MAE', 10, 4),
    'meld': ('MELD', 'classification', 'weight_F1', 8, 2),
    'cherma': ('CHERMA0723', 'classification', 'weight_F1', 8, 2),
}


def add_upstream(root):
    root = Path(root).resolve()
    for name in ('models/multiTask/CMCM.py', 'models/subNets/Textmodel.py',
                 'models/ChatGLM3/modeling_chatglm.py', 'data/load_data.py',
                 'config/config_regression.py', 'config/config_classification.py',
                 'utils/metricsTop.py'):
        if not (root / name).is_file():
            raise FileNotFoundError(f'--upstream-root must contain {name}')
    # Fail rather than silently use another package with generic official names.
    for name in ('models', 'data', 'config', 'utils'):
        module = sys.modules.get(name)
        if module is not None:
            origin = getattr(module, '__file__', None)
            locations = list(getattr(module, '__path__', []))
            if origin:
                locations.append(origin)
            if not locations or not all(Path(p).resolve().is_relative_to(root) for p in locations):
                raise RuntimeError(f'Conflicting imported upstream package: {name}')
    sys.path.insert(0, str(root))


def configure(dataset, seed, model_path, data_root, device):
    relative, mode, metric, patience, budget = DATASETS[dataset]
    initial = SimpleNamespace(root_dataset_dir=str(Path(data_root).resolve()),
        modelName='cmcm', datasetName=dataset, train_mode=mode,
        pretrain_LM=str(Path(model_path).resolve()), num_workers=0, gpu_ids=[])
    name = 'regression' if mode == 'regression' else 'classification'
    cls = getattr(importlib.import_module(f'config.config_{name}'),
                  'ConfigRegression' if mode == 'regression' else 'ConfigClassification')
    args = cls(initial).get_config()
    # Some upstream revisions hardcode resource paths. Never edit their files.
    args.dataPath = str(Path(data_root).resolve() / relative)
    args.root_dataset_dir = str(Path(data_root).resolve())
    args.pretrain_LM = str(Path(model_path).resolve())
    args.device = torch.device(device)
    args.seed = seed
    args.learning_rate = 5e-5
    if (args.batch_size, args.update_epochs, args.pseudo_tokens,
            args.KeyEval, args.early_stop, args.max_new_tokens) != (8, 1, 4, metric, patience, budget):
        raise ValueError('External configuration differs from the supported protocol')
    return args


class OfficialRuntime:
    def __init__(self, upstream_root):
        add_upstream(upstream_root)

    def load_split(self, args, split):
        dataset_cls = importlib.import_module('data.load_data').MMDataset
        data = dataset_cls(args, mode=split)
        for values, lengths in ((data.audio, data.audio_lengths),
                                (data.vision, data.vision_lengths)):
            lengths = np.asarray(lengths)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite features after official preprocessing')
            if (lengths < 1).any() or (lengths > values.shape[1]).any():
                raise ValueError('Invalid declared sequence lengths')
        if args.train_mode == 'classification' and not set(data.labels['M']) <= set(range(7)):
            raise ValueError('Unknown class label')
        return data

    def make_model(self, args, variant):
        cls = importlib.import_module('models.multiTask.CMCM').CMCM
        model = cls(args)
        shared_rng = rng_state()
        install_adapter(model, variant)
        restore_rng(shared_rng)
        model.to(args.device)
        if any(p.requires_grad for p in model.LLM.model.parameters()):
            raise ValueError('The LLM backbone must remain frozen')
        return model

    def metrics(self, args, predictions, truths):
        cls = importlib.import_module('utils.metricsTop').MetricsTop
        metric = cls(args).getMetics(args.datasetName)
        if args.train_mode == 'regression':
            return metric(torch.tensor(predictions), torch.tensor(truths))
        return metric(predictions, truths)
