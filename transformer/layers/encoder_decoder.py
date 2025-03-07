import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class EncoderDecoder(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, src_embed, tgt_embed, generator):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):

        encode = self.encode(src, src_mask) 

        return self.decode(encode, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)