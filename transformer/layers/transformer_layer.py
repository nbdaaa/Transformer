import torch
import torch.nn as nn
from position_wise_feed_forward_network import FeedForward
from multi_head_attention import MultiHeadAttention

class TransformerLayer(nn.Module):
    def __init__(self, *, d_model, self_attn : MultiHeadAttention, cross_attn, feed_forward : FeedForward, dropout_prob):
        super().__init__()

        self.size = d_model

        self.self_attn = self_attn

        self.cross_attn = cross_attn

        self.feed_forward = feed_forward

        self.dropout = nn.Dropout(dropout_prob)
        
        self.norm_self_attn = nn.LayerNorm([d_model])

        if self.cross_attn is not None:
            self.norm_cross_attn = nn.LayerNorm([d_model])

        self.norm_ff = nn.LayerNorm([d_model])

        self.is_save_ff_input = False

    def forward(self, *, x, mask, cross, cross_mask):
        z = self.norm_self_attn(x)

        z = self.self_attn(query=z, key=z, value=z, mask=mask)

        x += self.dropout(z)

        if cross is not None:
            z = self.norm_cross_attn(x)

            z = self.cross_attn(query=z, key=cross, value=cross, mask=cross_mask)

            x += self.dropout(z)

        z = self.norm_ff(x)

        if self.is_save_ff_input:
            self.ff_input = z.clone()

        ff = self.feed_forward(z)

        x += self.dropout(ff)        
 
        return x