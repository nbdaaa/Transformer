import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re
import os

def analyze_dataset(csv_file):
    """
    Analyze the news summarization dataset and print statistics.
    """
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Compute statistics on article and summary lengths
    df['article_length'] = df['Original'].apply(lambda x: len(str(x).split()))
    df['summary_length'] = df['Summary'].apply(lambda x: len(str(x).split()))
    df['compression_ratio'] = df['summary_length'] / df['article_length']
    
    print("\nArticle statistics:")
    print(f"Average word count: {df['article_length'].mean():.2f}")
    print(f"Median word count: {df['article_length'].median():.2f}")
    print(f"Min word count: {df['article_length'].min()}")
    print(f"Max word count: {df['article_length'].max()}")
    
    print("\nSummary statistics:")
    print(f"Average word count: {df['summary_length'].mean():.2f}")
    print(f"Median word count: {df['summary_length'].median():.2f}")
    print(f"Min word count: {df['summary_length'].min()}")
    print(f"Max word count: {df['summary_length'].max()}")
    
    print("\nCompression ratio statistics:")
    print(f"Average compression ratio: {df['compression_ratio'].mean():.4f}")
    print(f"Median compression ratio: {df['compression_ratio'].median():.4f}")
    
    # Create directory for plots
    os.makedirs("dataset_analysis", exist_ok=True)
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(df['article_length'], bins=50, alpha=0.7, label='Articles')
    plt.hist(df['summary_length'], bins=50, alpha=0.7, label='Summaries')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Article and Summary Lengths')
    plt.legend()
    plt.savefig('dataset_analysis/length_distribution.png')
    
    # Plot compression ratio
    plt.figure(figsize=(10, 6))
    plt.hist(df['compression_ratio'], bins=40, alpha=0.7)
    plt.xlabel('Compression Ratio (Summary Length / Article Length)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Compression Ratios')
    plt.savefig('dataset_analysis/compression_ratio.png')
    
    # Plot scatter plot of article vs summary length
    plt.figure(figsize=(10, 6))
    plt.scatter(df['article_length'], df['summary_length'], alpha=0.5)
    plt.xlabel('Article Length (words)')
    plt.ylabel('Summary Length (words)')
    plt.title('Relationship between Article and Summary Lengths')
    plt.savefig('dataset_analysis/length_relationship.png')
    
    print("\nAnalysis complete. Plots saved to 'dataset_analysis' directory.")

if __name__ == "__main__":
    analyze_dataset("all_news_summaries.csv")