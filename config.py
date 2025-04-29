from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 25,        # Increased for better convergence with long sequences
        "lr": 10**-4,
        "seq_len": 1024,         # For source articles
        "summary_len": 384,      # For target summaries
        "d_model": 512,
        "lang_src": "Original",
        "lang_tgt": "Summary",
        "model_folder": "weights",
        "model_basename": "summarizer_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/summarizer",
        "csv_file": "news_summaries_7.csv",
        "truncation_strategy": "start"  # Options: "start", "smart", or "chunk"
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