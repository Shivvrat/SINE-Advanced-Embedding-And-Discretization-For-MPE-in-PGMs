import math
import sys
import time

import torch
from torch import nn
from uai_reader import UAIParser


class BinaryBNModel(nn.Module):
    def __init__(
        self,
        uai_file,
        device,
    ) -> None:
        super().__init__()
        assert uai_file.endswith(".uai"), "Only support UAI format"
        self.pgm = UAIParser(uai_file, device)
        assert self.pgm.network_type == "BAYES", "Only support Markov Network"

    def evaluate(self, x):
        ll_scores = torch.zeros(x.shape[0], device=x.device)
        for table in self.pgm.prob_tables:
            func_vars, domains, cpd = table
            all_values = x[:, func_vars]

            # Precompute all possible binary combinations for the current table
            num_vars = len(func_vars)
            binary_combinations = torch.tensor(
                [
                    [(j >> k) & 1 for k in range(num_vars - 1, -1, -1)]
                    for j in range(2**num_vars)
                ],
                dtype=torch.float32,
                device=x.device,
            )

            # Compute the product terms using broadcasting
            expanded_values = all_values.unsqueeze(0).expand(2**num_vars, -1, -1)
            inverse_values = 1 - expanded_values
            selected_values = torch.where(
                binary_combinations.unsqueeze(1) == 1, expanded_values, inverse_values
            )
            product_terms = torch.prod(selected_values, dim=2).float()

            # Multiply by CPD values and sum over all combinations for each data point
            ll_scores += torch.matmul(cpd, product_terms)

        return ll_scores

    def evaluate_base(self, x):
        ll_scores = torch.zeros(x.shape[0]).to(x.device)
        for table in self.pgm.prob_tables:
            func_vars, domains, cpd = table
            all_values = x[:, func_vars]
            # convert each binary row to an integer using their binary representation
            # and use it as an index to get the corresponding cpd value
            for i, row in enumerate(all_values):
                this_ll_score = 0
                for j, log_pot_val in enumerate(cpd):
                    # convert j (index in cpd) to a binary vector of size x.shape[1] (1d tensor) to get the corresponding x variables
                    binary_repr = bin(j)[2:].zfill(row.shape[0])
                    j_bin = torch.tensor(
                        list(map(int, list(binary_repr))),
                        dtype=torch.float32,
                        device=x.device,
                    )
                    # for each index i take value from row if j_bin[i] == 1 else take 1 - row[i] using tensor torch.where
                    # then take the sum of all values in the row
                    this_val = log_pot_val * torch.prod(
                        torch.where(j_bin == 1, row, 1 - row)
                    )
                    this_ll_score += this_val
                # check if this_ll_score is nan
                if torch.isnan(this_ll_score):
                    print("nan")
                ll_scores[i] += this_ll_score
            # ll_score += log_cpd_value.sum()
        return ll_scores

    def evaluate_two_loops(self, x):
        ll_scores = torch.zeros(x.shape[0], device=x.device)
        for table in self.pgm.prob_tables:
            func_vars, _, cpd = table
            all_values = x[:, func_vars]

            # Precompute all possible binary combinations for the current table
            num_vars = len(func_vars)
            binary_combinations = torch.tensor(
                [
                    [(j >> k) & 1 for k in range(num_vars - 1, -1, -1)]
                    for j in range(2**num_vars)
                ],
                dtype=torch.float32,
                device=x.device,
            )

            # Initialize product_terms
            product_terms = torch.empty((2**num_vars, x.shape[0]), device=x.device)

            # Compute the product for each combination
            for i, combination in enumerate(binary_combinations):
                term = torch.ones(x.shape[0], device=x.device)
                for var_idx in range(num_vars):
                    term *= torch.where(
                        combination[var_idx] == 1,
                        all_values[:, var_idx],
                        1 - all_values[:, var_idx],
                    )
                product_terms[i] = term

            # Multiply by CPD values and sum over all combinations for each data point
            cpd_values = torch.tensor(cpd, dtype=torch.float32, device=x.device)
            ll_scores += torch.matmul(cpd_values, product_terms)

        return ll_scores

    @torch.no_grad()
    def evaluate_no_grad(self, x):
        ll_scores = torch.zeros(x.shape[0], device=x.device)

        for table in self.pgm.prob_tables:
            func_vars, domains, cpd = table
            all_values = x[:, func_vars]

            # Ensure all_values is of type LongTensor
            all_values = all_values.float()

            # Convert binary rows to integer indices using bitwise operations
            # Ensure power-of-two vector is also of type LongTensor
            indices = all_values.matmul(
                2
                ** torch.arange(all_values.size(-1) - 1, -1, -1, dtype=torch.float)
                .to(x.device)
                .unsqueeze(0)
                .T
            )

            # Vectorized addition of cpd values to ll_scores
            ll_scores += cpd[indices.long()].squeeze()

        return ll_scores

    def evaluate_no_grad_seq(self, x):
        ll_scores = torch.zeros(x.shape[0]).to(x.device)
        for table in self.pgm.prob_tables:
            func_vars, domains, cpd = table
            all_values = x[:, func_vars]
            # convert each binary row to an integer using their binary representation
            # and use it as an index to get the corresponding cpd value
            for i, row in enumerate(all_values):
                row = [int(i) for i in row]
                index = int("".join(map(str, row)), 2)
                ll_scores[i] += cpd[index]

            # ll_score += log_cpd_value.sum()
        return ll_scores
