import re
import torch


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for i, char in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def state_dict(self):
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_state_dict(cls, state):
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = state["stoi"]
        tokenizer.itos = {int(k): v for k, v in state["itos"].items()}
        tokenizer.vocab_size = len(tokenizer.stoi)
        tokenizer.chars = sorted(tokenizer.stoi.keys())
        return tokenizer


# def preprocess_text(text):
#     text = re.sub(r"[^a-zA-Z0-9 .*\n]", "", text)
#     return text.lower()

def preprocess_text(text):
    text = text.lower()

    # Keep printable ASCII + newline
    text = ''.join(
        c for c in text
        if c in '\n\t' or 32 <= ord(c) <= 126
    )

    return text


def load_text(path, fraction=1.0):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the range (0, 1].")

    return text[:int(fraction * len(text))]


def prepare_data(path, fraction=1.0, train_split=0.9):
    raw_text = load_text(path, fraction)

    print(f"Original length: {len(raw_text)}")

    text = preprocess_text(raw_text)
    print(f"Cleaned length: {len(text)}")

    tokenizer = CharTokenizer(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(tokenizer.chars)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(train_split * len(data))
    train_data = data[:split]
    val_data = data[split:]

    print(f"Train characters: {len(train_data)}")
    print(f"Validation characters: {len(val_data)}")

    return text, tokenizer, train_data, val_data


def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size:
        raise ValueError("Dataset must be longer than block_size.")

    starts = torch.randint(0, len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])

    return x.to(device), y.to(device)
