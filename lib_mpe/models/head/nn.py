# nn.py

from typing import List

import torch
import torch.nn as nn
from lib_mpe.models.head.base_head import BaseHead
from lib_mpe.project_utils.profiling import pytorch_profile


class NeuralNetwork(BaseHead):
    def __init__(self, cfg, input_size: int, hidden_sizes: List[int], output_size: int):
        super(NeuralNetwork, self).__init__(cfg)
        self.num_hidden_layers = cfg.model_layers
        self._set_activation_func(cfg)

        last_hidden_layer_size = self.init_hidden_layers(
            cfg, input_size, hidden_sizes, output_size
        )

        if self.num_hidden_layers > -1:
            self.output_layer = nn.Linear(last_hidden_layer_size, output_size)
        elif self.num_hidden_layers == -1:
            self.output_layer = nn.Parameter(torch.randn(output_size))
            self.input_values = torch.ones(
                1, self.output_layer.shape[0], dtype=float, device=cfg.data_device
            )
        else:
            raise ValueError("Invalid number of hidden layers")

        self.ste = None
        self.initialize_weights()

    def forward(self, x):
        if self.num_hidden_layers > 0:
            for layer in self.hidden_layers:
                x = layer(x)
                if not self.no_dropout and isinstance(layer, nn.ReLU):
                    x = self.dropout(x)
            model_output = self.output_layer(x)
        elif self.num_hidden_layers == 0:
            model_output = self.output_layer(x)
        elif self.num_hidden_layers == -1:
            model_output = self.input_values * self.output_layer
        else:
            raise ValueError("Invalid number of hidden layers")
        return model_output

    def init_hidden_layers(self, cfg, input_size, hidden_sizes, output_size):
        layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(
                nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], hidden_size)
            )
            layers.append(self.hidden_activation())
            if not cfg.no_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))

        if self.num_hidden_layers == 0:
            last_hidden_layer_size = input_size
        elif self.num_hidden_layers == -1:
            last_hidden_layer_size = -1
        else:
            self.hidden_layers = nn.Sequential(*layers)
            last_hidden_layer_size = hidden_sizes[-1]

        return last_hidden_layer_size

    def initialize_weights(self):
        if self.num_hidden_layers > 0:
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_in",
                        nonlinearity=self.hidden_activation_function,
                    )
                    nn.init.zeros_(layer.bias)
        if self.num_hidden_layers > -1:
            nn.init.xavier_uniform_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def get_features(self, x, layer_index):
        i = 0
        for layer in self.hidden_layers:
            x = layer(x)
            if i == layer_index:
                return x
            if isinstance(layer, nn.Linear):
                i += 1
        return x
