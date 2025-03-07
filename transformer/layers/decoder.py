import torch
import torch.nn as nn

from transformer_layer import TransformerLayer

class Decoder(nn.Module):
    def __init__(self, layer : TransformerLayer, n_layers):
        super().__init__()
        
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])

        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x, mask, cross, cross_mask):
        for layer in self.layers:
            x = layer(x=x, mask=mask, cross=cross, cross_mask=cross_mask)

        return self.norm(x)