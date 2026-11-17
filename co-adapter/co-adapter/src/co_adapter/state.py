"""Small atomic state helpers. Only load checkpoints from trusted runs."""
import json
import os
import random
import uuid

import numpy as np
import torch

PROCESS = uuid.uuid4().hex


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(path.read_text())


def save_checkpoint(path, value):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def read_checkpoint(path, identity):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['identity'] != identity:
        raise ValueError('Checkpoint run identity differs')
    return value


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def model_state(model):
    # Frozen weights are intentionally external, not duplicated in every checkpoint.
    return {'parameters': {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                           if p.requires_grad},
            'buffers': {n: b.detach().cpu().clone() for n, b in model.named_buffers()}}


def restore_model(model, saved):
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    if expected.keys() != saved['parameters'].keys() or buffers.keys() != saved['buffers'].keys():
        raise ValueError('Checkpoint architecture differs')
    with torch.no_grad():
        for current, values in ((expected, saved['parameters']), (buffers, saved['buffers'])):
            for name, value in values.items():
                if current[name].shape != value.shape or current[name].dtype != value.dtype:
                    raise ValueError(f'Checkpoint tensor differs: {name}')
                current[name].copy_(value)
