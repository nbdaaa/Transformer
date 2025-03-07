import math
import torch
from torch import nn as nn

class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model, heads, d_k, bias):
        super().__init__()
        
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        
        self.heads = heads
        
        self.d_k = d_k
    
    def forward(self, x):
        head_shape = x.shape[:-1]
        
        x = self.linear(x)

        x = x.view(*head_shape, self.heads, self.d_k)

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout_prob, bias):
        super().__init__()

        self.d_k = d_model // heads

        self.heads = heads

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim = 1)

        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_prob)

        self.scale = 1 / (self.d_k)**(1/2)

        self.attn = None

    def get_scores(self, query, key):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask, query_shape, key_shape):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)

        return mask

    def forward(self, *, query, key, value, mask):
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)    
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)

        attn = self.dropout(attn)

        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)