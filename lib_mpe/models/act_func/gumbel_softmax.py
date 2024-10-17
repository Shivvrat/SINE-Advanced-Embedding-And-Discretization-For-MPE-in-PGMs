import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSigmoid(nn.Module):
    def __init__(
        self, initial_temperature=5.0, min_temperature=0.5, annealing_rate=0.99
    ):
        super(GumbelSigmoid, self).__init__()
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.annealing_rate = annealing_rate

    def forward(self, logits, hard=False):
        ret = gumbel_sigmoid(logits, temperature=self.temperature, hard=hard)
        # Anneal temperature
        self.temperature = max(
            self.temperature * self.annealing_rate, self.min_temperature
        )
        return ret


def gumbel_sigmoid(logits, temperature=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = torch.sigmoid(gumbels)

    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
