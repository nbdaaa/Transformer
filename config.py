from pathlib import Path
import torch

def get_config():
    return {
        "batch_size": 16,           # Increased for GPU efficiency (adjust based on GPU memory)
        "num_epochs": 25,
        "lr": 10**-4,
        "seq_len": 1024,            # For source articles
        "summary_len": 384,         # For target summaries
        "d_model": 512,
        "lang_src": "Original",     # Column name for source text
        "lang_tgt": "Summary",      # Column name for target text
        "model_folder": "weights",
        "model_basename": "summarizer_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/summarizer",
        "csv_file": "news_summaries_7.csv",
        "truncation_strategy": "start",  # Options: "start", "smart", "middle", "end"
        "gpu_id": 0,                # GPU ID to use (0 for first GPU)
        "mixed_precision": True,    # Use mixed precision for faster training
        "gradient_accumulation_steps": 1,  # Increase this if GPU memory is limited
        "checkpoint_every_n_steps": 5000,  # Save checkpoints during training
        "max_grad_norm": 1.0        # Gradient clipping norm
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"summarizer_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"summarizer_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

# Get the device based on available hardware and config
def get_device(config):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config['gpu_id']}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        
        # Set GPU memory usage strategy
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
        return device
    else:
        print("WARNING: No GPU found, using CPU. Training will be much slower.")
        return torch.device("cpu")