import torch
import torch.nn as nn
from tokenizers import Tokenizer
import sys
import os
from pathlib import Path
import argparse
import time
from torch.cuda.amp import autocast

# Add parent directory to path to import from other modules
sys.path.append('.')

from dataset import causal_mask
from model import build_transformer
from config import get_config, latest_weights_file_path, get_device

def summarize(article_text, beam_search=False, beam_size=5):
    """
    Generate a summary for the provided article using the trained transformer model.
    
    Args:
        article_text: The input article text to summarize
        beam_search: Whether to use beam search decoding (slower but better quality)
        beam_size: Beam size for beam search decoding
        
    Returns:
        A string containing the generated summary
    """
    start_time = time.time()
    
    # Get configuration
    config = get_config()
    
    # Set device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Load tokenizers
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        raise Exception("Tokenizer files not found. Please train the model first.")
    
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    
    # Load the trained model
    model_path = latest_weights_file_path(config)
    if model_path is None:
        raise Exception("Model weights not found. Please train the model first.")
    
    print(f"Loading model from {model_path}")
    
    # Get sequence lengths from config
    src_seq_len = config.get('seq_len', 1024)
    tgt_seq_len = config.get('summary_len', 384)
    
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        src_seq_len,
        tgt_seq_len,
        d_model=config['d_model']
    ).to(device)
    
    # Load the pretrained weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    
    # Tokenize the input text
    sos_token_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_token_id = tokenizer_tgt.token_to_id('[EOS]')
    pad_token_id = tokenizer_tgt.token_to_id('[PAD]')
    
    # Process the article text
    tokens = tokenizer_src.encode(article_text).ids
    
    # Get truncation strategy from config
    truncation_strategy = config.get('truncation_strategy', 'start')
    
    # Apply truncation if the input is too long
    if len(tokens) > src_seq_len - 2:  # -2 for SOS and EOS
        if truncation_strategy == "start":
            tokens = tokens[:src_seq_len - 2]
        elif truncation_strategy == "end":
            tokens = tokens[-(src_seq_len - 2):]
        elif truncation_strategy == "smart":
            # Take first 75% and last 25% of tokens
            first_part = int(0.75 * (src_seq_len - 2))
            last_part = (src_seq_len - 2) - first_part
            tokens = tokens[:first_part] + tokens[-last_part:]
        elif truncation_strategy == "middle":
            # Take first and last parts equally
            half_len = (src_seq_len - 2) // 2
            tokens = tokens[:half_len] + tokens[-half_len:]
    
    # Prepare the input tensor
    enc_input = torch.cat([
        torch.tensor([sos_token_id], dtype=torch.int64),
        torch.tensor(tokens, dtype=torch.int64),
        torch.tensor([eos_token_id], dtype=torch.int64),
        torch.tensor([pad_token_id] * (src_seq_len - len(tokens) - 2), dtype=torch.int64)
    ]).unsqueeze(0).to(device)
    
    # Create mask
    enc_mask = (enc_input != pad_token_id).unsqueeze(0).unsqueeze(0).int().to(device)
    
    # Generate the summary
    with torch.no_grad():
        if beam_search:
            summary = beam_search_decode(
                model, enc_input, enc_mask, sos_token_id, eos_token_id, 
                pad_token_id, tokenizer_tgt, tgt_seq_len, device, beam_size
            )
        else:
            summary = greedy_decode(
                model, enc_input, enc_mask, sos_token_id, eos_token_id, 
                pad_token_id, tokenizer_tgt, tgt_seq_len, device
            )
    
    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    
    return summary

def greedy_decode(model, enc_input, enc_mask, sos_token_id, eos_token_id, pad_token_id, tokenizer_tgt, max_len, device):
    """
    Greedy decoding strategy - selects the most probable next token at each step.
    """
    # Initialize the decoder input with the SOS token
    dec_input = torch.empty(1, 1).fill_(sos_token_id).type_as(enc_input).to(device)
    
    # Encode the input sequence
    with autocast(enabled=torch.cuda.is_available()):
        enc_output = model.encode(enc_input, enc_mask)
    
    # Generate tokens one by one
    while dec_input.size(1) < max_len:
        # Create causal mask for decoder
        dec_mask = causal_mask(dec_input.size(1)).type_as(enc_mask).to(device)
        
        # Get decoder output
        with autocast(enabled=torch.cuda.is_available()):
            dec_output = model.decode(enc_output, enc_mask, dec_input, dec_mask)
            # Get the next token probabilities
            prob = model.project(dec_output[:, -1])
        
        # Get the token with highest probability
        _, next_word = torch.max(prob, dim=1)
        
        # Add the new token to decoder input
        dec_input = torch.cat([
            dec_input,
            torch.empty(1, 1).type_as(enc_input).fill_(next_word.item()).to(device)
        ], dim=1)
        
        # Stop if end of sequence
        if next_word == eos_token_id:
            break
    
    # Convert token ids back to text
    summary_tokens = dec_input.squeeze(0).detach().cpu().numpy().tolist()
    summary = tokenizer_tgt.decode(summary_tokens)
    
    # Clean up the summary
    summary = summary.replace("[SOS]", "").replace("[EOS]", "").strip()
    return summary

def beam_search_decode(model, enc_input, enc_mask, sos_token_id, eos_token_id, 
                     pad_token_id, tokenizer_tgt, max_len, device, beam_size=5):
    """
    Beam search decoding - maintains top-k hypotheses at each step.
    Produces higher quality summaries but is slower than greedy decoding.
    """
    # Precompute encoder output once
    with autocast(enabled=torch.cuda.is_available()):
        encoder_output = model.encode(enc_input, enc_mask)
    
    # Start with a single beam containing just the SOS token
    beams = [(torch.tensor([[sos_token_id]], device=device), 0)]  # (sequence, score)
    completed_beams = []
    
    for _ in range(max_len):
        new_beams = []
        
        # Expand each current beam
        for sequence, score in beams:
            # If the sequence is completed, add to completed beams
            if sequence[0, -1].item() == eos_token_id:
                completed_beams.append((sequence, score))
                continue
                
            # Create mask for the decoder
            decoder_mask = causal_mask(sequence.size(1)).type_as(enc_mask).to(device)
            
            # Get predictions for next token
            with autocast(enabled=torch.cuda.is_available()):
                decoder_output = model.decode(encoder_output, enc_mask, sequence, decoder_mask)
                next_token_logits = model.project(decoder_output[:, -1])
            
            # Get top-k tokens and their probabilities
            topk_probs, topk_ids = torch.topk(torch.softmax(next_token_logits, dim=1), beam_size)
            
            # Add each candidate to new beams
            for i in range(beam_size):
                token_id = topk_ids[0, i].item()
                token_prob = topk_probs[0, i].item()
                
                # Create new sequence by adding this token
                new_sequence = torch.cat([sequence, torch.tensor([[token_id]], device=device)], dim=1)
                
                # Update score (log probability)
                new_score = score + torch.log(topk_probs[0, i]).item()
                
                new_beams.append((new_sequence, new_score))
        
        # Keep only the top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Early stopping if all beams have generated EOS
        if all(beam[0][0, -1].item() == eos_token_id for beam in beams):
            break
    
    # Add any unfinished beams to the completed list
    completed_beams.extend(beams)
    
    # Sort by score and return the best sequence
    if completed_beams:
        best_sequence, _ = max(completed_beams, key=lambda x: x[1])
        summary_tokens = best_sequence.squeeze(0).detach().cpu().numpy().tolist()
    else:
        # Fallback if no beam was completed
        summary_tokens = beams[0][0].squeeze(0).detach().cpu().numpy().tolist()
    
    # Convert tokens to text
    summary = tokenizer_tgt.decode(summary_tokens)
    
    # Clean up the summary
    summary = summary.replace("[SOS]", "").replace("[EOS]", "").strip()
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate summary from an article")
    parser.add_argument("--article", type=str, help="Article text or file path to summarize")
    parser.add_argument("--file", action="store_true", help="Indicates that the article argument is a file path")
    parser.add_argument("--beam", action="store_true", help="Use beam search decoding (higher quality but slower)")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search decoding")
    parser.add_argument("--output", type=str, help="Output file to save the generated summary")
    
    args = parser.parse_args()
    
    if args.article:
        # Get the article text
        if args.file:
            # Read from file
            try:
                with open(args.article, 'r', encoding='utf-8') as f:
                    article_text = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
        else:
            article_text = args.article
        
        # Generate summary
        try:
            summary = summarize(article_text, args.beam, args.beam_size)
            print("\nGENERATED SUMMARY:")
            print("="*50)
            print(summary)
            print("="*50)
            
            # Save to output file if specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary saved to {args.output}")
                
        except Exception as e:
            print(f"Error generating summary: {e}")
            sys.exit(1)
    else:
        print("Please provide an article to summarize using the --article argument")
        sys.exit(1)