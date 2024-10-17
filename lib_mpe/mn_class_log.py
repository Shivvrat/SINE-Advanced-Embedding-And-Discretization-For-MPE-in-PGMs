import math
import time

import numpy as np
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from torch import nn

from lib_mpe.reader.uai_reader_cython import UAIParser


class BinaryMNModel(nn.Module):
    def __init__(
        self,
        uai_file,
        device,
        debug=False,
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        assert uai_file.endswith(".uai"), "Only support UAI format"
        with open(uai_file, "r") as file:
            file_content = file.read()
        self.pgm = UAIParser(
            model_str=file_content, one_d_factors=0, device=self.device
        )
        assert self.pgm.network_type == "MARKOV", "Only support Markov Network"
        if self.pgm.pairwise_only:
            self.evaluate = self.evaluate_grids
        else:
            self.evaluate = self.evaluate_parallel_clique_dict
            self.precompute()

    @torch.no_grad()
    def initialize_evidence_cython(self, data: np.ndarray, evidence_bool: np.ndarray):
        """
        Initializes the evidence for the model by updating factors for each example.

        Parameters:
        - data: A binary numpy array of shape (n, d), where n is the number of examples and d is the number of variables.
        - evidence_bool: A binary numpy array of shape (n, d), where n is the number of examples and d is the number of variables.

        Returns:
        - instantiated_factors: A tensor of shape (n, num_cliques, 2^clique_size), where n is the number of examples, num_cliques is the number of cliques, and 2^clique_size is the number of possible values for the clique.
        - instantiated_vars: A tensor of shape (n, num_cliques, num_vars), where n is the number of examples, num_cliques is the number of cliques, and num_vars is the number of variables in the clique.
        - instantiated_domains: A tensor of shape (n, num_cliques, num_vars), where n is the number of examples, num_cliques is the number of cliques, and num_vars is the number of variables in the clique.
        """
        # a single tensor for all the factor values (values not used are initalized as 0 for factors and -1 for vars and domains)
        (
            instantiated_factors,
            instantiated_hyperedge_index,
            instantiated_vars,
            instantiated_domains,
        ) = self.pgm.instantiate_evidence_same_size_cliques_np(
            data.astype(np.int_).copy(), evidence_bool.copy()
        )
        return (
            instantiated_factors,
            instantiated_hyperedge_index,
            instantiated_vars,
            instantiated_domains,
        )

    def precompute(self):
        self.precomputed_data = {}
        for size, clique_class in self.pgm.clique_dict_class.items():
            binary_combinations = torch.tensor(
                [
                    [(j >> k) & 1 for k in range(size - 1, -1, -1)]
                    for j in range(2**size)
                ],
                dtype=torch.float32,
                device=self.device,
            )
            all_vars = clique_class.variables
            all_factors = clique_class.tables
            self.precomputed_data[size] = {
                "binary_combinations": binary_combinations,
                "all_vars": all_vars,
                "all_factors": all_factors,
            }

    def compute_clique_scores(self, x, binary_combinations, all_vars, all_factors):
        all_values = x[:, all_vars.flatten()].view(
            x.shape[0], all_vars.shape[0], all_vars.shape[1]
        )

        selected_values = all_values.unsqueeze(1) * binary_combinations.unsqueeze(
            0
        ).unsqueeze(2) + (1 - all_values.unsqueeze(1)) * (
            1 - binary_combinations.unsqueeze(0).unsqueeze(2)
        )

        product_term = torch.prod(selected_values, dim=3)

        all_factors = all_factors.view(all_factors.shape[0], -1)

        scores = torch.sum(product_term * all_factors.permute(1, 0).unsqueeze(0), dim=1)

        return scores

    def evaluate_parallel_clique_dict(self, x):
        x = x.to(self.device)
        ll_scores = torch.zeros(x.shape[0], device=self.device)

        for size, data in self.precomputed_data.items():
            clique_scores = self.compute_clique_scores(
                x, data["binary_combinations"], data["all_vars"], data["all_factors"]
            )
            ll_scores += torch.sum(clique_scores, dim=1)

        return ll_scores

    def evaluate_grids(self, x):
        univariate_weights_0 = self.pgm.univariate_tables[:, 0]
        univariate_weights_1 = self.pgm.univariate_tables[:, 1]
        bivariate_weights_00 = self.pgm.bivariate_tables[:, 0, 0]
        bivariate_weights_01 = self.pgm.bivariate_tables[:, 0, 1]
        bivariate_weights_10 = self.pgm.bivariate_tables[:, 1, 0]
        bivariate_weights_11 = self.pgm.bivariate_tables[:, 1, 1]

        univariate_contributions = (
            1 - x[:, self.pgm.univariate_vars]
        ) * univariate_weights_0 + x[:, self.pgm.univariate_vars] * univariate_weights_1
        bivariate_contributions = (
            (1 - x[:, self.pgm.bivariate_vars[:, 0]])
            * (1 - x[:, self.pgm.bivariate_vars[:, 1]])
            * bivariate_weights_00.unsqueeze(0)
            + (1 - x[:, self.pgm.bivariate_vars[:, 0]])
            * x[:, self.pgm.bivariate_vars[:, 1]]
            * bivariate_weights_01.unsqueeze(0)
            + x[:, self.pgm.bivariate_vars[:, 0]]
            * (1 - x[:, self.pgm.bivariate_vars[:, 1]])
            * bivariate_weights_10.unsqueeze(0)
            + x[:, self.pgm.bivariate_vars[:, 0]]
            * x[:, self.pgm.bivariate_vars[:, 1]]
            * bivariate_weights_11.unsqueeze(0)
        )
        loss_val = torch.sum(univariate_contributions, dim=1) + torch.sum(
            bivariate_contributions, dim=1
        )

        return loss_val
