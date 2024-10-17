# transformer.py

from typing import List

import torch
import torch.nn as nn
from lib_mpe.models.head.base_head import BaseHead
from lib_mpe.models.head.pos_enc import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from lib_mpe.project_utils.profiling import pytorch_profile


class TransformerEncoder(BaseHead):
    def __init__(
        self,
        cfg,
        dropout: float,
        output_size: int,
        output_layers_sizes: int,
        positional_encoding: bool = True,
    ):
        super(TransformerEncoder, self).__init__(cfg)
        self.output_layers_sizes = output_layers_sizes
        self._set_activation_func(cfg)

        # Transformer Encoder Parameters
        d_model = (
            cfg.transformer_d_model if hasattr(cfg, "transformer_d_model") else None
        )
        nhead = cfg.transformer_nhead if hasattr(cfg, "transformer_nhead") else 4
        num_layers = (
            cfg.transformer_num_layers if hasattr(cfg, "transformer_num_layers") else 2
        )
        dim_feedforward = (
            cfg.transformer_dim_feedforward
            if hasattr(cfg, "transformer_dim_feedforward")
            else 2048
        )
        dropout = cfg.dropout_rate if hasattr(cfg, "dropout_rate") else 0.1
        max_seq_length = (
            cfg.transformer_max_seq_length
            if hasattr(cfg, "transformer_max_seq_length")
            else 100
        )
        self.d_model = d_model

        if positional_encoding:
            self.pos_encoder = self._get_positional_encoding(
                cfg, d_model, max_seq_length
            )
        else:
            self.pos_encoder = None

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=cfg.hidden_activation_function,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        self.set_output_layers(cfg, d_model)
        self.initialize_weights()

    def forward(self, x):
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        if not self.no_dropout and self.dropout is not None:
            transformer_output = self.dropout(transformer_output)
        model_output = self.output_layers(transformer_output)
        model_output = model_output.squeeze(-1)
        return model_output

    def set_output_layers(self, cfg, d_model):
        output_layers = []
        for i, hidden_size in enumerate(self.output_layers_sizes):
            output_layers.append(
                nn.Linear(
                    d_model if i == 0 else self.output_layers_sizes[i - 1], hidden_size
                )
            )
            output_layers.append(self.hidden_activation())
            if not cfg.no_batchnorm:
                output_layers.append(nn.BatchNorm1d(hidden_size))
        output_layers.append(nn.Linear(self.output_layers_sizes[-1], 1))
        if len(self.output_layers_sizes) > 0:
            self.output_layers = nn.Sequential(*output_layers)

    def initialize_weights(self):
        for name, module in self.transformer_encoder.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for name, module in self.output_layers.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_positional_encoding(self, cfg, d_model, max_seq_length):
        if cfg.positional_encoding == "sinusoidal":
            return SinusoidalPositionalEncoding(
                d_model, max_len=max_seq_length, batch_first=True
            )
        elif cfg.positional_encoding == "learned":
            return LearnedPositionalEncoding(
                d_model, max_len=max_seq_length, batch_first=True
            )
        else:
            raise ValueError(
                f"Unsupported positional encoding: {cfg.positional_encoding}"
            )
