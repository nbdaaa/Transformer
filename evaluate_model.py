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

# Add parent directory to path to import from other modules
sys.path.append('.')

from model import build_transformer
from dataset import SummarizationDataset, causal_mask
from config import get_config, latest_weights_file_path
from tokenizers import Tokenizer

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
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

def evaluate_model(test_file=None, num_samples=None, model_path=None):
    """
    Evaluate the model on a test dataset
    
    Args:
        test_file: CSV file containing test data (if None, uses part of the training data)
        num_samples: Number of samples to evaluate (if None, evaluates all)
        model_path: Path to the model weights file (if None, uses latest)
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
    
    # Load the model weights
    if model_path is None:
        model_path = latest_weights_file_path(config)
        if model_path is None:
            raise Exception("Model weights not found. Please train the model first.")
    
    print(f"Loading model from {model_path}")
    
    # Create the model
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
    
    # Load test data
    if test_file is None:
        # Use part of the training data for evaluation
        test_file = config['csv_file']
    
    df = pd.read_csv(test_file)
    
    # If num_samples is provided, select a random subset
    if num_samples is not None and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)
    
    # Create dataset and dataloader
    test_ds = SummarizationDataset(df, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Initialize ROUGE
    rouge = Rouge()
    
    # Lists to store results
    source_texts = []
    reference_summaries = []
    generated_summaries = []
    rouge_scores = []
    
    # Evaluate the model
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Get inputs
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Generate summary
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
            
            # Decode the output
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            output_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            # Clean up the generated summary
            output_text = output_text.replace("[SOS]", "").replace("[EOS]", "").strip()
            
            # Store results
            source_texts.append(source_text)
            reference_summaries.append(target_text)
            generated_summaries.append(output_text)
            
            # Calculate ROUGE scores
            try:
                scores = rouge.get_scores(output_text, target_text)[0]
                rouge_scores.append(scores)
            except Exception as e:
                print(f"Error calculating ROUGE for a sample: {e}")
    
    # Calculate average ROUGE scores
    avg_rouge_1 = np.mean([score['rouge-1']['f'] for score in rouge_scores])
    avg_rouge_2 = np.mean([score['rouge-2']['f'] for score in rouge_scores])
    avg_rouge_l = np.mean([score['rouge-l']['f'] for score in rouge_scores])
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Number of samples evaluated: {len(source_texts)}")
    print(f"Average ROUGE-1 F1: {avg_rouge_1:.4f}")
    print(f"Average ROUGE-2 F1: {avg_rouge_2:.4f}")
    print(f"Average ROUGE-L F1: {avg_rouge_l:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Source': source_texts,
        'Reference': reference_summaries,
        'Generated': generated_summaries,
        'ROUGE-1 F1': [score['rouge-1']['f'] for score in rouge_scores],
        'ROUGE-2 F1': [score['rouge-2']['f'] for score in rouge_scores],
        'ROUGE-L F1': [score['rouge-l']['f'] for score in rouge_scores],
    })
    
    results_file = "evaluation_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # Display some examples
    print("\nExamples:")
    for i in range(min(3, len(source_texts))):
        print("-" * 80)
        print(f"ARTICLE: {source_texts[i][:200]}...")
        print(f"REFERENCE: {reference_summaries[i]}")
        print(f"GENERATED: {generated_summaries[i]}")
        print(f"ROUGE-1 F1: {rouge_scores[i]['rouge-1']['f']:.4f}")
    
    return avg_rouge_1, avg_rouge_2, avg_rouge_l

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate text summarization model")
    parser.add_argument("--test_file", type=str, help="CSV file containing test data")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--model_path", type=str, help="Path to the model weights file")
    
    args = parser.parse_args()
    
    # Evaluate the model
    evaluate_model(args.test_file, args.num_samples, args.model_path)