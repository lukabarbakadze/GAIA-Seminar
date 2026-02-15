"""
Visualization utilities for the Transformer seminar.
These are pre-written helper functions for plotting and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_training_curve(losses, title="Training Loss"):
    """Plot training loss over iterations."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses['train'], label='Train Loss', linewidth=2)
    plt.plot(losses['val'], label='Val Loss', linewidth=2)

    # Add random baseline
    random_loss = -np.log(1.0/49)  # vocab_size = 49
    plt.axhline(y=random_loss, color='r', linestyle='--', 
                label=f'Random Baseline ({random_loss:.2f})', linewidth=1)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_attention_heatmap(attention_weights, tokens=None, title="Attention Weights"):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: (T, T) tensor of attention weights
        tokens: Optional list of token strings for labels
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    # Convert to numpy if tensor
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()

    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')

    if tokens is not None:
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)

    plt.xlabel('Key Position', fontsize=12)
    plt.ylabel('Query Position', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_gpt_vs_bert_attention(model, text, decode_fn, device='cpu'):
    """
    Compare GPT (causal) vs BERT (bidirectional) attention patterns.
    
    Args:
        model: The GPTLanguageModel instance
        text: Input text string
        decode_fn: Function to decode token IDs to characters
        device: Device to run on
    """
    # Get attention weights from the model
    # This assumes the model has been modified to return attention weights
    # For demo purposes, we'll create synthetic examples

    T = min(len(text), 20)  # Limit to 20 tokens for visibility

    # Create synthetic attention patterns
    # GPT: Lower triangular (causal)
    gpt_attn = torch.tril(torch.ones(T, T))
    gpt_attn = gpt_attn / gpt_attn.sum(dim=-1, keepdim=True)

    # BERT: Full attention (bidirectional)
    bert_attn = torch.ones(T, T) / T

    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # GPT attention
    im1 = ax1.imshow(gpt_attn.numpy(), cmap='viridis', aspect='auto')
    ax1.set_title('GPT (Causal/Decoder)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1, label='Attention Weight')

    # BERT attention
    im2 = ax2.imshow(bert_attn.numpy(), cmap='viridis', aspect='auto')
    ax2.set_title('BERT (Bidirectional/Encoder)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    plt.colorbar(im2, ax=ax2, label='Attention Weight')

    plt.tight_layout()
    plt.show()

    print("\n🔑 Key Difference:")
    print("  GPT: Lower triangular → each token only sees PAST tokens (causal)")
    print("  BERT: Full matrix → each token sees ALL tokens (bidirectional)")


def plot_character_frequency(
    chars,
    text
):
    """Plot character frequency distribution."""
    from collections import Counter

    char_counts = Counter(text)
    sorted_chars = sorted(chars, key=lambda x: char_counts[x], reverse=True)
    counts = [char_counts[c] for c in sorted_chars]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(sorted_chars)), counts, color='steelblue', alpha=0.7)
    plt.xlabel('Character (sorted by frequency)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Character Frequency Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_chars)), 
               [c if c not in ['\n', ' '] else repr(c) for c in sorted_chars],
               rotation=45, ha='right', fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    print(f"\nTop 10 most frequent characters:")
    for i, (char, count) in enumerate(zip(sorted_chars[:10], counts[:10])):
        char_repr = repr(char) if char in ['\n', ' '] else char
        print(f"  {i+1}. '{char_repr}': {count:,} occurrences")


def visualize_embeddings_2d(
    embeddings,
    labels=None,
    title="Token Embeddings (2D Projection)"
):
    """
    Visualize high-dimensional embeddings in 2D using PCA.

    Args:
        embeddings: (vocab_size, n_embd) tensor
        labels: Optional list of labels for each embedding
        title: Plot title
    """
    from sklearn.decomposition import PCA

    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()

    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)

    if labels is not None:
        for i, label in enumerate(labels):
            if i % 3 == 0:  # Only label every 3rd point to avoid clutter
                plt.annotate(
                    label,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )

    plt.xlabel(
        f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
        fontsize=12
    )
    plt.ylabel(
        f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
        fontsize=12
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_generation_comparison(
    prompts,
    model,
    encode_fn,
    decode_fn,
    temperatures=[0.5, 1.0, 1.5],
    max_new_tokens=100
):
    """
    Generate text with different temperatures and compare.

    Args:
        prompts: List of prompt strings
        model: GPTLanguageModel instance
        encode_fn: Function to encode text to token IDs
        decode_fn: Function to decode token IDs to text
        temperatures: List of temperature values to try
        max_new_tokens: Number of tokens to generate
    """
    model.eval()

    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*80}")

        for temp in temperatures:
            context = torch.tensor([encode_fn(prompt)], dtype=torch.long)
            generated = model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=temp
            )
            output = decode_fn(generated[0].tolist())

            print(f"\n🌡️  Temperature = {temp}")
            print(f"{'─'*80}")
            print(output)
            print(f"{'─'*80}")


def plot_loss_landscape_2d(
    model,
    data_batch,
    param_names=['ln_f.weight', 'lm_head.weight'],
    steps=20,
    epsilon=0.1
):
    """
    Visualize loss landscape around current parameters (2D slice).
    Warning: This is computationally expensive!

    Args:
        model: The model to analyze
        data_batch: (X, Y) tuple of input and target tensors
        param_names: Names of two parameters to vary
        steps: Number of steps in each direction
        epsilon: Step size for parameter perturbation
    """
    X, Y = data_batch

    # Get the two parameters
    params = dict(model.named_parameters())
    param1 = params[param_names[0]]
    param2 = params[param_names[1]]

    # Store original values
    orig1 = param1.data.clone()
    orig2 = param2.data.clone()

    # Create grid
    losses = np.zeros((steps, steps))

    for i in range(steps):
        for j in range(steps):
            # Perturb parameters
            alpha = (i - steps//2) * epsilon
            beta = (j - steps//2) * epsilon

            param1.data = orig1 + alpha * torch.randn_like(orig1)
            param2.data = orig2 + beta * torch.randn_like(orig2)

            # Compute loss
            with torch.no_grad():
                _, loss = model(X, Y)
                losses[i, j] = loss.item()

    # Restore original parameters
    param1.data = orig1
    param2.data = orig2

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(losses, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel(param_names[1], fontsize=12)
    plt.ylabel(param_names[0], fontsize=12)
    plt.title('Loss Landscape (2D Slice)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
