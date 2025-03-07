import torch
import torch.nn as nn

from transformer_layer import TransformerLayer

class Generator(nn.Module):
    def __init__(self, n_vocab, d_model):
        super().__init__()
        self.pred_prob = nn.Linear(d_model, n_vocab)
    
    def forward(self, x):
        return self.pred_prob(x)