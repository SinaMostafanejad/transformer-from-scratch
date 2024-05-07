import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

    # def forward(self, x: torch.Tensor) -> torch.Tensor: return
    # self.embedding(x) * torch.sqrt(torch.tensor(self.d_model,
    # dtype=torch.float32))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in a matrix of shape (seq_len,
        # d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of positions (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Compute the positional encodings for the even indices using sin
        # function
        pe[:, 0::2] = torch.sin(position * div_term)

        # Compute the positional encodings for the odd indices using cos
        # function
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encodings
        pe = pe.unsqueeze(0)

        # Register the positional encodings as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps

        # nn.Parameter makes the multiplicative parameter trainable
        self.alpha = nn.Parameter(torch.ones(1))

        # nn.Parameter makes the additive parameter trainable
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # First linear layer with W_1 and b_1 parameters
        self.fc1 = nn.Linear(d_model, d_ff)

        # Second linear layer with W_2 and b_2 parameters
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Apply the first linear layer (batch, seq_len, d_model) -> (batch,
        # seq_len, d_ff)
        x = self.fc1(x)
        # Apply ReLU activation function
        x = torch.relu(x)
        # Apply dropout
        x = self.dropout(x)
        # Apply the second linear layer (batch, seq_len, d_ff) -> (batch,
        # seq_len, d_model)
        x = self.fc2(x)
        return x

