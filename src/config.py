from dataclasses import dataclass
import torch


@dataclass
class Config:
    data_path: str = "data/input.txt"
    model_dir: str = "models"
    checkpoint_name: str = "mini_gpt_checkpoint.pth"
    best_model_name: str = "mini_gpt_best.pth"

    data_fraction: float = 1
    train_split: float = 0.90

    block_size: int = 128
    embedding_dim: int = 320
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.0

    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 100

    seed: int = 42

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def checkpoint_path(self):
        return f"{self.model_dir}/{self.checkpoint_name}"

    @property
    def best_model_path(self):
        return f"{self.model_dir}/{self.best_model_name}"
