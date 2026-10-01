import torch
import torch.nn as nn

from .data import get_batch
from .utils import save_checkpoint


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    loss_fn,
    batch_size,
    block_size,
    device,
    eval_iters
):
    model.eval()

    results = {}

    for split_name, data_source in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(data_source, batch_size, block_size, device)

            logits = model(x)
            B, T, C = logits.shape

            loss = loss_fn(logits.reshape(B * T, C), y.reshape(B * T))
            losses[k] = loss.item()

        results[split_name] = losses.mean().item()

    model.train()
    return results


def train_model(model, train_data, val_data, config, tokenizer):
    device = config.device
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    model.train()

    for iteration in range(1, config.max_iters + 1):
        x, y = get_batch(train_data, config.batch_size, config.block_size, device)

        logits = model(x)
        B, T, C = logits.shape

        loss = loss_fn(logits.reshape(B * T, C), y.reshape(B * T))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % config.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                loss_fn,
                config.batch_size,
                config.block_size,
                device,
                config.eval_iters
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Iteration {iteration:5d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    config.best_model_path,
                    model,
                    config,
                    tokenizer,
                    optimizer,
                    iteration,
                    best_val_loss
                )

                print(
                    f"  Saved best model -> "
                    f"{config.best_model_path}"
                )

    save_checkpoint(
        config.checkpoint_path,
        model,
        config,
        tokenizer,
        optimizer,
        config.max_iters,
        best_val_loss
    )

    print(f"Final checkpoint saved -> {config.checkpoint_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss
    }