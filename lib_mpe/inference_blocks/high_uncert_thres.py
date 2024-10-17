import itertools

import torch


class HighUncertaintyVariableBasedThresholder:
    def __init__(self, k, scoring_function, num_query_variables, device, threshold):
        """
        Args:
            k (int): The number of query variables to select based on proximity to 0.5.
            scoring_function (function): A function that takes binary assignments and produces a score.
            num_query_variables (int): Number of query variables for binary assignment generation.
        """
        self.k = k
        self.scoring_function = scoring_function
        self.num_query_variables = num_query_variables
        self.threshold = threshold

        # Generate all possible binary assignments for the k most uncertain variables
        all_binary_assignments = list(itertools.product([0, 1], repeat=self.k))
        self.all_binary_assignments = torch.tensor(
            all_binary_assignments, dtype=torch.double, device=device
        )

    @torch.no_grad()
    def select_high_uncertainty(self, all_outputs_for_pgm, query_bool):
        num_examples = all_outputs_for_pgm.size(0)
        query_indices = (
            torch.where(torch.tensor(query_bool))[1]
            .reshape(num_examples, -1)
            .to(all_outputs_for_pgm.device)
        )
        final_assignments = all_outputs_for_pgm.clone()
        num_examples_better_than_threshold = 0

        for i in range(num_examples):
            example_output = all_outputs_for_pgm[i].unsqueeze(0)
            example_query_indices = query_indices[i]

            # Get the sigmoid outputs for the query variables
            query_probs = example_output[:, example_query_indices]

            # Measure uncertainty by getting absolute difference from 0.5
            uncertainty_scores = torch.abs(query_probs - 0.5)

            # Select the k most uncertain variables
            top_k_values, top_k_indices = torch.topk(
                uncertainty_scores, self.k, dim=1, largest=False
            )

            # Threshold the data
            thresholded_data = example_output.clone()
            thresholded_data[thresholded_data > self.threshold] = 1.0
            thresholded_data[thresholded_data <= self.threshold] = 0.0

            # Create all possible assignments for the top k uncertain variables
            num_assignments = len(self.all_binary_assignments)
            reshaped_data = thresholded_data.repeat(num_assignments, 1)

            # Update the top k uncertain variables with all possible binary assignments
            query_indices_to_update = example_query_indices[top_k_indices[0]]
            reshaped_data[:, query_indices_to_update] = self.all_binary_assignments

            # Score all assignments
            scores = self.scoring_function(reshaped_data)
            thresholded_score = self.scoring_function(thresholded_data)

            # Select the best scoring binary assignment
            best_assignment_index = torch.argmax(scores)

            # Update the final assignments
            if scores[best_assignment_index] > thresholded_score:
                final_assignments[i, query_indices_to_update] = (
                    self.all_binary_assignments[best_assignment_index]
                )
                num_examples_better_than_threshold += 1
            else:
                final_assignments[i, query_indices_to_update] = thresholded_data[
                    0, query_indices_to_update
                ]

        print(
            f"Number of examples better than threshold: {num_examples_better_than_threshold}/{num_examples}"
        )
        return final_assignments
