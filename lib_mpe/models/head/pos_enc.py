import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, batch_first: bool = False):
        """
        Implements the sinusoidal positional encoding.

        :param d_model: The dimension of the embeddings.
        :param max_len: The maximum length of sequences.
        :param batch_first: If True, input is expected to be of shape [batch_size, seq_len, d_model].
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.batch_first = batch_first

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Store pe without unsqueezing
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        :param x: Tensor of shape [batch_size, seq_len, d_model] if batch_first=True,
                  else [seq_len, batch_size, d_model]
        :return: Tensor with positional encoding added.
        """
        if self.batch_first:
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]
        else:
            seq_len = x.size(0)
            return x + self.pe[:seq_len]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, batch_first: bool = False):
        super(LearnedPositionalEncoding, self).__init__()
        self.batch_first = batch_first
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            seq_len = x.size(1)
            return x + self.positional_encoding[:, :seq_len, :]
        else:
            seq_len = x.size(0)
            return x + self.positional_encoding[:seq_len]
