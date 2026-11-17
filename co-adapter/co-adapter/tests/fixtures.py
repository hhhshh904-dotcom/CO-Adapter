"""Tiny deterministic data and a frozen differentiable LM for CPU tests only."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch import nn
from torch.utils.data import Dataset

from co_adapter.adapter import install_adapter
from co_adapter.state import rng_state, restore_rng


class Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def config(mode='regression'):
    return Config(datasetName='simsv2' if mode == 'regression' else 'meld',
                  train_mode=mode, device=torch.device('cpu'), seed=1111,
                  batch_size=8, seq_lens=(3, 4, 4), early_stop=1,
                  KeyEval='MAE' if mode == 'regression' else 'weight_F1',
                  max_new_tokens=4 if mode == 'regression' else 2)


class TinyData(Dataset):
    def __init__(self, split, mode):
        self.split, self.mode = split, mode
        self.ids = [f'{split}-{i}' for i in range(len(self))]

    def __len__(self):
        return 64 if self.split == 'train' else 16

    def get_seq_len(self):
        return (3, 4, 4)

    def __getitem__(self, index):
        ids = torch.tensor([index % 7, (index + 1) % 7, 0])
        text = torch.stack((ids, torch.tensor([1, 1, 0]), torch.zeros(3)))
        values = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12 + index / 64
        label = torch.tensor([float(index % 3) - 1]) if self.mode == 'regression' else index % 7
        return {'text': text, 'audio': values, 'vision': values / 2,
                'text_lengths': 3, 'audio_lengths': 1 + index % 4,
                'vision_lengths': 1 + (index + 1) % 4,
                'labels': {'M': label}, 'index': index}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(3, 4, batch_first=True)
        self.dropout = nn.Dropout(.1)
        self.linear = nn.Linear(4, 256)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 1)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.calls = []

    def stream_generate(self, fusion, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get('use_cache') is not False:
            raise AssertionError('Generation used cache')
        yield fusion


class TinyTokenizer:
    def batch_decode(self, outputs):
        return [str(float(x)) for x in outputs]


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.tokenizer = TinyTokenizer()
        self.dropout = nn.Dropout(.2)

    def text_embedding(self, ids):
        return self.model.embedding(ids)

    def forward(self, fusion, labels):
        output = self.model.head(self.dropout(fusion).mean(1)).view(-1)
        return SimpleNamespace(loss=(output - labels.float().view(-1)).square().mean())

    def generate(self, fusion):
        for output in self.model.stream_generate(fusion):
            values = self.model.head(output.mean(1)).view(-1).tolist()
            responses = self.tokenizer.batch_decode(values)
        return [float(x) for x in responses]


class TinyCMCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.LLM = TinyLLM()
        self.audio_LSTM = TinyEncoder()
        self.video_LSTM = TinyEncoder()
        # Preserve temporary allocations before installing the independent path.
        self.text_guide_mixer = nn.Linear(8, 8)
        self.mutli_scale_fusion = nn.Linear(8, 8)


class Runtime:
    def __init__(self, record=None):
        self.loaded = []
        self.record = record

    def load_split(self, args, split):
        self.loaded.append(split)
        if self.record:
            with open(self.record, 'a') as stream:
                stream.write(split + '\n')
        return TinyData(split, args.train_mode)

    def make_model(self, args, variant):
        model = TinyCMCM()
        state = rng_state()
        install_adapter(model, variant, hidden_size=8, width=8)
        restore_rng(state)
        return model

    def metrics(self, args, predictions, truths):
        # Stable selection exercises patience and first-best semantics.
        return {args.KeyEval: .5}


if __name__ == '__main__':
    from co_adapter import engine
    torch.set_num_threads(1)
    command, directory, variant, mode = sys.argv[1:5]
    out = Path(directory)
    runtime = Runtime(out / 'access.txt')
    args = config(mode)
    resources = {'fixture': 'tiny-v1'}
    if command in ('smoke', 'resume-smoke', 'interrupt'):
        engine.train(args, variant, out / 'smoke', runtime, resources,
                     resume=command == 'resume-smoke', smoke=True,
                     stop_after_batches=3 if command == 'interrupt' else None)
    elif command == 'reload-smoke':
        engine.replay_and_evaluate(args, variant, out / 'smoke', runtime, resources, smoke=True)
    elif command == 'train':
        engine.train(args, variant, out, runtime, resources)
    elif command == 'evaluate':
        engine.replay_and_evaluate(args, variant, out, runtime, resources)
    else:
        raise ValueError(command)
