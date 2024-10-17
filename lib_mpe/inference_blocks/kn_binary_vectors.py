import numpy as np
import torch
from tqdm import tqdm

from lib_mpe.inference_blocks.cython_code.kn_binary_vectors import (
    cython_process_assignments,
)


class KNearestBinaryVectorsFinder:
    def __init__(
        self, k, scoring_function, num_query_variables, device, batch_size=300
    ):
        """
        Args:
            k (int): The number of nearest binary vectors to find.
            scoring_function (function): A function that takes binary assignments and produces a score.
            num_query_variables (int): Number of query variables for binary assignment generation.
            device: torch.device on which tensors will be allocated.
        """
        self.k = k
        self.scoring_function = scoring_function
        self.num_query_variables = num_query_variables
        self.device = device
        self.batch_size = k

    @torch.no_grad()
    def select_k_nearest(self, all_outputs_for_pgm, query_bool):
        """
        Args:
            all_outputs_for_pgm (torch.Tensor): Tensor of continuous outputs (batch_size, num_variables).
            query_bool (torch.Tensor): Boolean mask indicating which variables are query variables.

        Returns:
            torch.Tensor: Tensor of the same shape as `all_outputs_for_pgm` with updated query variable values.
        """
        num_examples, num_vars = all_outputs_for_pgm.shape
        # Get indices of query variables
        query_indices = (
            torch.where(torch.tensor(query_bool))[1]
            .reshape(num_examples, -1)
            .to(all_outputs_for_pgm.device)
        )

        # For each example, get the continuous outputs for the query variables
        s = all_outputs_for_pgm[
            torch.arange(num_examples)[:, None], query_indices
        ].view(num_examples, -1)
        s_np = s.cpu().numpy()

        # Initialize the final assignments tensor
        final_assignments = all_outputs_for_pgm.clone()

        best_assignments = all_outputs_for_pgm.clone()
        best_scores = torch.empty(num_examples, device=self.device)
        for i in tqdm(range(num_examples), desc="Finding K-Nearest Binary Vectors"):
            s_i = s_np[
                i
            ]  # Continuous outputs for this example (size: num_query_variables)
            N = s_i.shape[0]

            # Initialize L as a list of tuples: (total_distance, assignment)
            L, np_assignments = cython_process_assignments(s_i, self.k)

            # Convert assignments to tensor
            assignments = torch.tensor(
                np_assignments,
                dtype=torch.double,
                device=self.device,
            )
            # assignments shape: (num_assignments, N)

            # query_indices[i] is the indices of query variables for this example
            num_assignments = assignments.shape[0]
            query_indices_i = query_indices[i]  # Shape: (num_query_variables,)
            # Process assignments in smaller batches
            batch_size = self.batch_size
            best_score = float("-inf")
            best_assignment = None

            for j in range(0, num_assignments, batch_size):
                batch = assignments[j : j + batch_size]
                data_batch = final_assignments[i].unsqueeze(0).repeat(len(batch), 1)
                data_batch[:, query_indices_i] = batch

                scores = self.scoring_function(data_batch).view(-1)
                max_score, max_idx = torch.max(scores, dim=0)

                if max_score > best_score:
                    best_score = max_score
                    best_assignment = batch[max_idx]

            best_assignments[i, query_indices_i] = best_assignment
            best_scores[i] = best_score

        # Compute thresholded scores for all examples at once
        thresholded_data = (all_outputs_for_pgm >= 0.5).float()
        thresholded_scores = self.scoring_function(thresholded_data)
        # best_assignment_scores = self.scoring_function(best_assignments)
        final_assignments = torch.where(
            best_scores[:, np.newaxis] > thresholded_scores[:, np.newaxis],
            best_assignments,
            thresholded_data,
        )

        num_examples_better_than_threshold = torch.sum(best_scores > thresholded_scores)
        print(
            f"Number of examples better than threshold: {num_examples_better_than_threshold}/{num_examples}"
        )
        number_examples_equal_to_threshold = torch.sum(
            best_scores == thresholded_scores
        )
        print(
            f"Number of examples equal to threshold: {number_examples_equal_to_threshold}/{num_examples}"
        )

        return final_assignments
