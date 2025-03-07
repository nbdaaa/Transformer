import torch
from torch import nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_prob, activation = nn.ReLU()):
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob) 

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)

        return self.layer2(x)
