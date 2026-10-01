import math
import os
import re
import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import preprocess_text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.03)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.03)


def count_parameters(model):
    total = 0
    breakdown = {}

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            count = parameter.numel()
            total += count
            breakdown[name] = count

    summary = {
        "total_parameters": total,
        "total_parameters_millions": total / 1e6,
        "breakdown": breakdown
    }

    print(f"Total trainable parameters: {total:,}")
    print(f"Total parameters in millions: {total / 1e6:.2f}M")
    print("\n" + "-" * 75)

    for name, count in breakdown.items():
        print(f"{name:60s} {count:10,}")

    return summary


def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MiniGPT Training")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_checkpoint(
    path,
    model,
    config,
    tokenizer,
    optimizer=None,
    iteration=None,
    best_val_loss=None
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": vars(config),
        "tokenizer": tokenizer.state_dict(),
        "iteration": iteration,
        "best_val_loss": best_val_loss
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model_class, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["config"]
    tokenizer_state = checkpoint["tokenizer"]

    model = model_class(
        vocab_size=len(tokenizer_state["stoi"]),
        block_size=config["block_size"],
        embedding_dim=config["embedding_dim"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def perplexity(loss):
    return math.exp(loss)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    block_size,
    device,
    max_new_tokens,
    temperature=1.0,
    top_k=None
):
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    prompt = preprocess_text(prompt)

    unknown = sorted(set(prompt) - set(tokenizer.stoi))
    if unknown:
        raise ValueError(
            f"Prompt contains characters not in vocabulary: {unknown}"
        )

    idx = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_recent = idx[:, -block_size:]

        logits = model(idx_recent)
        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, k)

            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )
            filtered_logits.scatter_(1, indices, values)
            logits = filtered_logits

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0].tolist())


def plot_embedding_pca(model, tokenizer):
    embeddings = (
        model.embedding.token_embedding.weight
        .detach()
        .cpu()
        .numpy()
    )

    print("Embedding shape:", embeddings.shape)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    print("PCA shape:", embeddings_2d.shape)
    print(
        f"Explained variance: "
        f"{pca.explained_variance_ratio_}"
    )
    print(
        f"Total explained variance: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    plt.figure(figsize=(12, 8))

    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1]
    )

    for i, char in enumerate(tokenizer.chars):
        plt.annotate(
            repr(char),
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Learned Character Embeddings")
    plt.grid(alpha=0.2)
    plt.show()