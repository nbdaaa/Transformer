import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
from rouge import Rouge
import time
import json
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

# Add parent directory to path to import from other modules
sys.path.append('.')

from model import build_transformer
from dataset import SummarizationDataset, causal_mask
from config import get_config, latest_weights_file_path, get_device
from tokenizers import Tokenizer

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    with autocast(enabled=torch.cuda.is_available()):
        encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        with autocast(enabled=torch.cuda.is_available()):
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            # get next token
            prob = model.project(out[:, -1])
        
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=5):
    """
    More advanced beam search decoding for better quality summaries.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Precompute encoder output once
    with autocast(enabled=torch.cuda.is_available()):
        encoder_output = model.encode(source, source_mask)
    
    # Start with a single beam containing just the SOS token
    beams = [(torch.tensor([[sos_idx]], device=device), 0)]  # (sequence, score)
    completed_beams = []
    
    for _ in range(max_len):
        new_beams = []
        
        # Expand each current beam
        for sequence, score in beams:
            # If the sequence is completed, add to completed beams
            if sequence[0, -1].item() == eos_idx:
                completed_beams.append((sequence, score))
                continue
                
            # Create mask for the decoder
            decoder_mask = causal_mask(sequence.size(1)).type_as(source_mask).to(device)
            
            # Get predictions for next token
            with autocast(enabled=torch.cuda.is_available()):
                decoder_output = model.decode(encoder_output, source_mask, sequence, decoder_mask)
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
                new_score = score + np.log(token_prob)
                
                new_beams.append((new_sequence, new_score))
        
        # Keep only the top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Early stopping if all beams have generated EOS
        if all(beam[0][0, -1].item() == eos_idx for beam in beams):
            break
    
    # Add any unfinished beams to the completed list
    completed_beams.extend(beams)
    
    # Sort by score and return the best sequence
    if completed_beams:
        best_sequence, _ = max(completed_beams, key=lambda x: x[1])
        return best_sequence.squeeze(0)
    else:
        # Fallback if no beam was completed
        return beams[0][0].squeeze(0)

def evaluate_model(test_file=None, num_samples=None, model_path=None, beam_search=False, beam_size=5, output_file=None):
    """
    Evaluate the model on a test dataset
    
    Args:
        test_file: CSV file containing test data (if None, uses part of the training data)
        num_samples: Number of samples to evaluate (if None, evaluates all)
        model_path: Path to the model weights file (if None, uses latest)
        beam_search: Whether to use beam search decoding
        beam_size: Beam size for beam search decoding
        output_file: Output file for evaluation results
    """
    start_time = time.time()
    
    # Get configuration
    config = get_config()
    
    # Set device
    device = get_device(config)
    
    # Load tokenizers
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))
    
    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        raise Exception("Tokenizer files not found. Please train the model first.")
    
    print(f"Loading tokenizers from {tokenizer_src_path} and {tokenizer_tgt_path}")
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    
    # Load the model weights
    if model_path is None:
        model_path = latest_weights_file_path(config)
        if model_path is None:
            raise Exception("Model weights not found. Please train the model first.")
    
    print(f"Loading model from {model_path}")
    
    # Create the model
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
    
    # Load test data
    if test_file is None:
        # Use part of the training data for evaluation
        test_file = config['csv_file']
    
    print(f"Loading test data from {test_file}")
    df = pd.read_csv(test_file)
    
    # If num_samples is provided, select a random subset
    if num_samples is not None and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
    
    # Create dataset and dataloader
    truncation_strategy = config.get('truncation_strategy', 'start')
    test_ds = SummarizationDataset(
        df, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], 
        src_seq_len, tgt_seq_len, truncation_strategy
    )
    
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True)
    
    # Initialize ROUGE
    rouge = Rouge()
    
    # Lists to store results
    results = {
        'samples': [],
        'metrics': {
            'rouge-1': {'f': 0, 'p': 0, 'r': 0},
            'rouge-2': {'f': 0, 'p': 0, 'r': 0},
            'rouge-l': {'f': 0, 'p': 0, 'r': 0}
        },
        'time_taken': 0,
        'config': {
            'model_path': model_path,
            'beam_search': beam_search,
            'beam_size': beam_size if beam_search else None,
            'test_file': test_file,
            'num_samples': len(df)
        }
    }
    
    # Evaluation progress bar
    progress_bar = tqdm(total=len(test_dataloader), desc="Evaluating")
    
    # Evaluate the model
    with torch.no_grad():
        for batch in test_dataloader:
            # Get inputs
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Start generation timer
            gen_start = time.time()
            
            # Generate summary using appropriate decoding method
            if beam_search:
                model_out = beam_search_decode(
                    model, encoder_input, encoder_mask, 
                    tokenizer_src, tokenizer_tgt, 
                    tgt_seq_len, device, beam_size
                )
            else:
                model_out = greedy_decode(
                    model, encoder_input, encoder_mask, 
                    tokenizer_src, tokenizer_tgt, 
                    tgt_seq_len, device
                )
            
            # Record generation time
            generation_time = time.time() - gen_start
            
            # Get source and target texts
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            
            # Decode the output
            output_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            output_text = output_text.replace("[SOS]", "").replace("[EOS]", "").strip()
            
            # Calculate ROUGE scores
            try:
                scores = rouge.get_scores(output_text, target_text)[0]
                
                # Store sample results
                sample_result = {
                    'source': source_text[:200] + "..." if len(source_text) > 200 else source_text,
                    'reference': target_text,
                    'generated': output_text,
                    'generation_time_sec': generation_time,
                    'rouge_scores': scores
                }
                
                results['samples'].append(sample_result)
                
                # Update total metrics (to be averaged later)
                for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                    for score_type in ['f', 'p', 'r']:
                        results['metrics'][metric][score_type] += scores[metric][score_type]
                
            except Exception as e:
                print(f"Error calculating ROUGE for a sample: {e}")
                
            # Update progress bar
            progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    # Calculate average metrics
    sample_count = len(results['samples'])
    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
        for score_type in ['f', 'p', 'r']:
            results['metrics'][metric][score_type] /= sample_count
    
    # Record total time
    total_time = time.time() - start_time
    results['time_taken'] = total_time
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Number of samples evaluated: {sample_count}")
    print(f"Decoding method: {'Beam Search (beam_size=' + str(beam_size) + ')' if beam_search else 'Greedy'}")
    print(f"ROUGE-1 F1: {results['metrics']['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {results['metrics']['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {results['metrics']['rouge-l']['f']:.4f}")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    
    # Save results to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")
    
    # Display visual sample
    print("\nSample Results:")
    for i in range(min(3, len(results['samples']))):
        sample = results['samples'][i]
        print("-" * 80)
        print(f"ARTICLE: {sample['source']}")
        print(f"REFERENCE: {sample['reference']}")
        print(f"GENERATED: {sample['generated']}")
        print(f"ROUGE-1 F1: {sample['rouge_scores']['rouge-1']['f']:.4f}")
        print(f"ROUGE-L F1: {sample['rouge_scores']['rouge-l']['f']:.4f}")
    
    # Plot ROUGE scores distribution
    plt.figure(figsize=(10, 6))
    rouge1_scores = [s['rouge_scores']['rouge-1']['f'] for s in results['samples']]
    rouge2_scores = [s['rouge_scores']['rouge-2']['f'] for s in results['samples']]
    rougel_scores = [s['rouge_scores']['rouge-l']['f'] for s in results['samples']]
    
    plt.hist(rouge1_scores, alpha=0.7, label='ROUGE-1 F1', bins=20)
    plt.hist(rouge2_scores, alpha=0.7, label='ROUGE-2 F1', bins=20)
    plt.hist(rougel_scores, alpha=0.7, label='ROUGE-L F1', bins=20)
    
    plt.xlabel('ROUGE Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of ROUGE Scores')
    plt.legend()
    
    plot_file = f"rouge_scores_{'beam' if beam_search else 'greedy'}.png"
    plt.savefig(plot_file)
    print(f"ROUGE scores distribution plot saved to {plot_file}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate text summarization model")
    parser.add_argument("--test_file", type=str, help="CSV file containing test data")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--model_path", type=str, help="Path to the model weights file")
    parser.add_argument("--beam_search", action="store_true", help="Use beam search decoding")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search decoding")
    parser.add_argument("--output", type=str, help="Output file for evaluation results")
    
    args = parser.parse_args()
    
    # Evaluate the model
    evaluate_model(
        args.test_file, 
        args.num_samples, 
        args.model_path,
        args.beam_search,
        args.beam_size,
        args.output
    )