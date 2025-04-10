from math import pi, log

import torch
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from beartype import beartype
from beartype.typing import Literal, Union, Optional
from rotary_embedding_torch import RotaryEmbedding


class CustomRotaryEmbedding(RotaryEmbedding):
    @autocast(enabled=False)
    def forward(
        self,
        t: Tensor,
        seq_len=None,
        offset=0
    ):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel'
        )

        if (
            should_cache and
            exists(self.cached_freqs) and
            (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        # 将 freqs 移动到输入张量 t 所在的设备
        freqs = self.freqs.to(t.device)

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs
