"""Four modality-local input tokens, with optional mixed query conditioning."""
import math
import types

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_mean(values, mask):
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


class MaskedAttention(nn.Module):
    def __init__(self, width=128, memory_dim=256, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(memory_dim, width)
        self.value = nn.Linear(memory_dim, width)
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(memory_dim)

    def forward(self, query, memory, mask):
        batch, count, _ = query.shape
        steps = memory.shape[1]
        q = self.query(self.query_norm(query.float()))
        memory = self.memory_norm(memory.float())
        k, v = self.key(memory), self.value(memory)

        def split(x, length):
            return x.reshape(batch, length, self.heads, self.width // self.heads).transpose(1, 2)

        q, k, v = split(q, count), split(k, steps), split(v, steps)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width // self.heads)
        weights = scores.masked_fill(~mask[:, None, None, :], -1e4).softmax(-1)
        weights = weights * mask[:, None, None, :].to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return (weights @ v).transpose(1, 2).reshape(batch, count, self.width)


class IndependentTokens(nn.Module):
    """Two learned queries per modality; audio tokens precede video tokens."""
    def __init__(self, hidden_size=4096, width=128, variant='co'):
        super().__init__()
        if variant not in ('co', 'mixed'):
            raise ValueError('Unknown query variant')
        self.variant = variant
        self.queries = nn.Parameter(torch.randn(2, 2, width) * 0.02)
        # Retained even in CO to preserve allocation and initialization order.
        self.text_query = nn.Linear(hidden_size, width)
        self.attention = nn.ModuleList([MaskedAttention(width) for _ in range(2)])
        self.output = nn.ModuleList([nn.Linear(width, hidden_size) for _ in range(2)])

    def forward(self, text, text_mask, memories):
        condition = (self.text_query(masked_mean(text.float(), text_mask))
                     if self.variant == 'mixed' else None)
        tokens = []
        for i, (memory, mask) in enumerate(memories):
            if condition is None:
                queries = self.queries[i].unsqueeze(0).expand(text.shape[0], -1, -1)
            else:
                content = self.queries[i, 0].expand(text.shape[0], -1)
                context = self.queries[i, 1].expand(text.shape[0], -1) + condition
                queries = torch.stack((content, context), dim=1)
            read = self.attention[i](queries, memory, mask)
            tokens.append(self.output[i](read) * mask.any(1)[:, None, None])
        return torch.cat(tokens, dim=1)


def temporal_memory(encoder, values, lengths):
    """Expose all valid states of the official LSTM, dropout and projection."""
    lengths = lengths.to(device=values.device, dtype=torch.long)
    if (lengths < 1).any() or (lengths > values.shape[1]).any():
        raise ValueError('Invalid official sequence lengths')
    packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
    sequence, _ = encoder.rnn(packed)
    sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=values.shape[1])
    mask = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
    memory = encoder.linear(encoder.dropout(sequence)).masked_fill(~mask[..., None], 0)
    return memory, mask


def install_adapter(model, variant='co', hidden_size=4096, width=128):
    """Install only after official CMCM construction, preserving its RNG draws."""
    del model.text_guide_mixer
    del model.mutli_scale_fusion
    model.independent = IndependentTokens(hidden_size, width, variant)

    def fuse(self, text, audio, video):
        text_tensor, _ = text
        embeddings = self.LLM.text_embedding(text_tensor[:, 0, :].long())
        memories = (temporal_memory(self.audio_LSTM, *audio),
                    temporal_memory(self.video_LSTM, *video))
        tokens = self.independent(embeddings, text_tensor[:, 1, :].bool(), memories)
        # Text padding remains part of the LLM input, as in the official code.
        return torch.cat((tokens, embeddings), dim=1), tokens

    def forward(self, labels, text, audio, video):
        fusion, tokens = fuse(self, text, audio, video)
        return {'Loss': self.LLM(fusion, labels).loss, 'Feature_f': tokens}

    def generate(self, text, audio, video):
        fusion, _ = fuse(self, text, audio, video)
        return self.LLM.generate(fusion)

    model.forward = types.MethodType(forward, model)
    model.generate = types.MethodType(generate, model)
    return model
