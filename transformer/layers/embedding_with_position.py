import math

import torch
import torch.nn as nn

from generates_position import get_positional_encoding

class EmbeddingsWithPositionalEncoding(nn.Module):
    def __init__(self, d_model, n_vocab, max_len):
        super().__init__()
        
        self.linear = nn.Embedding(n_vocab, d_model)
        
        self.d_model = d_model

        self.positional_encoding = get_positional_encoding(d_model, max_len)

    def forward(self, x):
        pe = self.positional_encoding[:x.shape[0], :, :].requires_grad_(False)

        return self.linear(x) * (self.d_model ** 0.5) + pe
    