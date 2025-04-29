import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class SummarizationDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_column, tgt_column, src_seq_len, tgt_seq_len, truncation_strategy="start"):
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.truncation_strategy = truncation_strategy

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_column = src_column
        self.tgt_column = tgt_column

        # Precompute special tokens for faster processing
        self.sos_token_id = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token_id = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
        
        self.sos_token = torch.tensor([self.sos_token_id], dtype=torch.int64)
        self.eos_token = torch.tensor([self.eos_token_id], dtype=torch.int64)
        self.pad_token = torch.tensor([self.pad_token_id], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds.iloc[idx]
        src_text = item[self.src_column]
        tgt_text = item[self.tgt_column]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Apply truncation strategy for source tokens if they're too long
        if len(enc_input_tokens) > self.src_seq_len - 2:  # -2 for SOS and EOS
            enc_input_tokens = self._apply_truncation(enc_input_tokens, self.src_seq_len - 2)
        
        # Apply truncation for target tokens if they're too long
        if len(dec_input_tokens) > self.tgt_seq_len - 2:  # -2 for SOS and EOS
            dec_input_tokens = dec_input_tokens[:self.tgt_seq_len - 2]  # Simple truncation for target

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.src_seq_len - len(enc_input_tokens) - 2  # For SOS and EOS
        dec_num_padding_tokens = self.tgt_seq_len - len(dec_input_tokens) - 1  # Only add SOS to decoder input

        # Create tensors efficiently using pre-allocated memory
        # Encoder input: [SOS] + tokens + [EOS] + padding
        encoder_input = torch.empty(self.src_seq_len, dtype=torch.int64)
        encoder_input[0] = self.sos_token_id
        encoder_input[1:len(enc_input_tokens)+1] = torch.tensor(enc_input_tokens, dtype=torch.int64)
        encoder_input[len(enc_input_tokens)+1] = self.eos_token_id
        if enc_num_padding_tokens > 0:
            encoder_input[len(enc_input_tokens)+2:] = self.pad_token_id

        # Decoder input: [SOS] + tokens + padding
        decoder_input = torch.empty(self.tgt_seq_len, dtype=torch.int64)
        decoder_input[0] = self.sos_token_id
        decoder_input[1:len(dec_input_tokens)+1] = torch.tensor(dec_input_tokens, dtype=torch.int64)
        if dec_num_padding_tokens > 0:
            decoder_input[len(dec_input_tokens)+1:] = self.pad_token_id

        # Label: tokens + [EOS] + padding
        label = torch.empty(self.tgt_seq_len, dtype=torch.int64)
        label[:len(dec_input_tokens)] = torch.tensor(dec_input_tokens, dtype=torch.int64)
        label[len(dec_input_tokens)] = self.eos_token_id
        if dec_num_padding_tokens > 0:
            label[len(dec_input_tokens)+1:] = self.pad_token_id

        # Create masks efficiently
        encoder_mask = (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token_id).unsqueeze(0).int() & causal_mask(self.tgt_seq_len)

        return {
            "encoder_input": encoder_input,  # (src_seq_len)
            "decoder_input": decoder_input,  # (tgt_seq_len)
            "encoder_mask": encoder_mask,    # (1, 1, src_seq_len)
            "decoder_mask": decoder_mask,    # (1, tgt_seq_len, tgt_seq_len)
            "label": label,                  # (tgt_seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
    def _apply_truncation(self, tokens, max_len):
        """Apply truncation strategy to tokens that exceed max_len"""
        if self.truncation_strategy == "start":
            # Take tokens from the beginning
            return tokens[:max_len]
        
        elif self.truncation_strategy == "end":
            # Take tokens from the end
            return tokens[-max_len:]
        
        elif self.truncation_strategy == "smart":
            # Take first 3/4 and last 1/4 of tokens
            first_part = int(0.75 * max_len)
            last_part = max_len - first_part
            return tokens[:first_part] + tokens[-last_part:]
        
        elif self.truncation_strategy == "middle":
            # Take first and last parts, skipping the middle
            half_len = max_len // 2
            return tokens[:half_len] + tokens[-half_len:]
        
        else:
            # Default to beginning truncation
            return tokens[:max_len]

# Create the causal mask once and cache it for reuse
_causal_masks = {}
    
def causal_mask(size):
    """
    Create a causal mask for the decoder with caching for better GPU performance.
    """
    if size in _causal_masks:
        return _causal_masks[size]
        
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    mask = mask == 0
    _causal_masks[size] = mask
    return mask