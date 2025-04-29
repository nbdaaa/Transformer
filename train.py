from model import build_transformer
from dataset import SummarizationDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path, get_device

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast

import warnings
from tqdm import tqdm
import os
import time
from pathlib import Path
import gc

# Tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from rouge import Rouge

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    with torch.cuda.amp.autocast(enabled=True):
        encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        with torch.cuda.amp.autocast(enabled=True):
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


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            model_out_text = model_out_text.replace("[SOS]", "").replace("[EOS]", "").strip()
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            # Truncate long articles for display
            if len(source_text) > 300:
                display_src = source_text[:150] + "..." + source_text[-150:]
            else:
                display_src = source_text
                
            print_msg(f"{f'ARTICLE: ':>12}{display_src}")
            print_msg(f"{f'REFERENCE: ':>12}{target_text}")
            print_msg(f"{f'GENERATED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Calculate ROUGE score
        if len(predicted) > 0 and len(expected) > 0:
            try:
                rouge = Rouge()
                scores = rouge.get_scores(predicted, expected, avg=True)
                writer.add_scalar('validation ROUGE-1 F1', scores['rouge-1']['f'], global_step)
                writer.add_scalar('validation ROUGE-2 F1', scores['rouge-2']['f'], global_step)
                writer.add_scalar('validation ROUGE-L F1', scores['rouge-l']['f'], global_step)
                writer.flush()
                
                # Print ROUGE scores
                print_msg(f"ROUGE-1 F1: {scores['rouge-1']['f']:.4f}")
                print_msg(f"ROUGE-2 F1: {scores['rouge-2']['f']:.4f}")
                print_msg(f"ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
            except Exception as e:
                print(f"Error calculating ROUGE: {e}")

        # Evaluate the character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(df, column):
    for item in df[column]:
        yield item

def get_or_build_tokenizer(config, df, column_name):
    tokenizer_path = Path(config['tokenizer_file'].format(column_name))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(df, column_name), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # Load the CSV file containing the news articles and summaries
    print(f"Loading dataset from {config['csv_file']}...")
    df = pd.read_csv(config['csv_file'])
    print(f"Dataset loaded with {len(df)} samples.")
    
    # Check if columns exist
    if config['lang_src'] not in df.columns or config['lang_tgt'] not in df.columns:
        raise ValueError(f"CSV file must contain '{config['lang_src']}' and '{config['lang_tgt']}' columns")
    
    # Build tokenizers
    print("Building tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, df, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, df, config['lang_tgt'])
    print("Tokenizers ready.")

    # Keep 90% for training, 10% for validation
    train_size = int(0.9 * len(df))
    val_size = len(df) - train_size
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Get the sequence lengths (with defaults if not specified)
    src_seq_len = config.get('seq_len', 1024)
    tgt_seq_len = config.get('summary_len', 384)
    truncation_strategy = config.get('truncation_strategy', 'start')

    print(f"Creating training dataset with {len(train_df)} samples...")
    train_ds = SummarizationDataset(
        train_df, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], 
        src_seq_len, tgt_seq_len, truncation_strategy
    )
    
    print(f"Creating validation dataset with {len(val_df)} samples...")
    val_ds = SummarizationDataset(
        val_df, tokenizer_src, tokenizer_tgt, 
        config['lang_src'], config['lang_tgt'], 
        src_seq_len, tgt_seq_len, truncation_strategy
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    print("Estimating token lengths from sample...")
    sample_size = min(1000, len(df))  # Sample to estimate lengths
    for _, row in df.sample(sample_size).iterrows():
        src_ids = tokenizer_src.encode(row[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(row[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Estimated max length of source articles: {max_len_src} tokens')
    print(f'Estimated max length of target summaries: {max_len_tgt} tokens')
    print(f'Using source sequence length: {src_seq_len}')
    print(f'Using target sequence length: {tgt_seq_len}')
    print(f'Truncation strategy: {truncation_strategy}')
    
    # Create data loaders with pin_memory for faster GPU transfer
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        pin_memory=True,
        num_workers=2,  # Parallel data loading
        prefetch_factor=2  # Prefetch batches
    )
    
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=True,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    # Get the sequence lengths from config
    src_seq_len = config.get('seq_len', 1024)
    tgt_seq_len = config.get('summary_len', 384)
    
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, 
        src_seq_len, tgt_seq_len, 
        d_model=config['d_model']
    )
    return model

def train_model(config):
    # Get device from config
    device = get_device(config)
    
    # Enable cuDNN autotuner to find the best algorithm
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Make sure the weights folder exists
    Path(f"summarizer_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Load datasets and create model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Print model parameters and size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # Setup gradient scaler for mixed precision training
    scaler = GradScaler(enabled=config.get('mixed_precision', False))

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
        # Move optimizer states to GPU if needed
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        print('No model to preload, starting from scratch')

    # Setup loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        label_smoothing=0.1
    ).to(device)

    # Get the target sequence length for validation
    tgt_seq_len = config.get('summary_len', 384)
    
    # Setup gradient accumulation
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    effective_batch_size = config['batch_size'] * gradient_accumulation_steps
    print(f"Training with effective batch size: {effective_batch_size}")

    # Start training
    print(f"Starting training from epoch {initial_epoch} to {config['num_epochs']}")
    for epoch in range(initial_epoch, config['num_epochs']):
        start_time = time.time()
        
        # Clear GPU cache before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        # Training loop
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device) 
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Use mixed precision training
            with autocast(enabled=config.get('mixed_precision', False)):
                # Forward pass
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                
                # Compute loss
                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                    label.view(-1)
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Accumulate loss for reporting
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients to prevent explosion
                if config.get('max_grad_norm', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0))
                
                # Update weights with scaled gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Update progress bar
                batch_iterator.set_postfix({
                    "loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to tensorboard
                writer.add_scalar('train/loss', loss.item() * gradient_accumulation_steps, global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Save checkpoint periodically
                if config.get('checkpoint_every_n_steps', 0) > 0 and global_step > 0 and global_step % config.get('checkpoint_every_n_steps') == 0:
                    checkpoint_filename = get_weights_file_path(config, f"{epoch:02d}_{global_step}")
                    print(f"\nSaving checkpoint to {checkpoint_filename}")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'loss': total_loss / (batch_idx + 1)
                    }, checkpoint_filename)
                
                global_step += 1

        # Log average loss for epoch
        avg_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s - Average Loss: {avg_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        
        # Run validation
        print("Running validation...")
        run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, 
            tgt_seq_len, device, 
            lambda msg: print(msg),  # Print directly instead of using batch_iterator 
            global_step, writer, num_examples=3
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        print(f"Saving model to {model_filename}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': avg_loss
        }, model_filename)
        
        # Clear any remaining GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)