import torch
import torch.nn as nn
from tokenizers import Tokenizer
import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path to import from other modules
sys.path.append('.')

from dataset import causal_mask
from model import build_transformer
from config import get_config, latest_weights_file_path

def summarize(article_text):
    """
    Generate a summary for the provided article using the trained transformer model.
    
    Args:
        article_text: The input article text to summarize
        
    Returns:
        A string containing the generated summary
    """
    # Get configuration
    config = get_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)
    
    # Load the pretrained weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    
    # Tokenize the input text
    sos_token = tokenizer_tgt.token_to_id('[SOS]')
    eos_token = tokenizer_tgt.token_to_id('[EOS]')
    pad_token = tokenizer_tgt.token_to_id('[PAD]')
    
    # Process the article text
    tokens = tokenizer_src.encode(article_text).ids
    
    # If the input is too long, truncate it
    if len(tokens) > config['seq_len'] - 2:  # -2 for SOS and EOS
        tokens = tokens[:config['seq_len'] - 2]
    
    # Prepare the input tensor
    enc_input = torch.cat([
        torch.tensor([sos_token], dtype=torch.int64),
        torch.tensor(tokens, dtype=torch.int64),
        torch.tensor([eos_token], dtype=torch.int64),
        torch.tensor([pad_token] * (config['seq_len'] - len(tokens) - 2), dtype=torch.int64)
    ]).unsqueeze(0).to(device)
    
    # Create mask
    enc_mask = (enc_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
    
    # Generate the summary
    with torch.no_grad():
        # Initialize the decoder input with the SOS token
        dec_input = torch.empty(1, 1).fill_(sos_token).type_as(enc_input).to(device)
        
        # Generate tokens one by one
        while dec_input.size(1) < config['seq_len']:
            # Create causal mask for decoder
            dec_mask = causal_mask(dec_input.size(1)).type_as(enc_mask).to(device)
            
            # Get encoder output
            enc_output = model.encode(enc_input, enc_mask)
            
            # Get decoder output
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
            if next_word == eos_token:
                break
        
        # Convert token ids back to text
        summary_tokens = dec_input.squeeze(0).detach().cpu().numpy().tolist()
        summary = tokenizer_tgt.decode(summary_tokens)
        
        # Clean up the summary (remove SOS and EOS tokens, if present)
        summary = summary.replace("[SOS]", "").replace("[EOS]", "").strip()
        return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate summary from an article")
    parser.add_argument("--article", type=str, help="Article text or file path to summarize")
    parser.add_argument("--file", action="store_true", help="Indicates that the article argument is a file path")
    
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
            summary = summarize(article_text)
            print("\nGENERATED SUMMARY:")
            print("="*50)
            print(summary)
            print("="*50)
        except Exception as e:
            print(f"Error generating summary: {e}")
            sys.exit(1)
    else:
        print("Please provide an article to summarize using the --article argument")
        sys.exit(1)