import os
import tempfile
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from lib_mpe.inference_blocks.inf_utils.b_and_b import run_branch_and_bound

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "30"


def process_example(i, all_outputs, query_idx, evid_idx, params):
    example_output = all_outputs[i]
    example_query_indices = query_idx[i]
    example_evid_indices = evid_idx[i]
    num_query_vars = params["num_query_variables"]
    i_bound = params["i_bound"]

    query_probs = example_output[example_query_indices]
    uncertainty_scores = np.abs(query_probs - 0.5)

    top_k_indices = np.argsort(uncertainty_scores)[: params["branch_bound_max_vars"]]
    all_query_indices = np.arange(num_query_vars)
    certain_indices = np.setdiff1d(all_query_indices, top_k_indices)
    example_evid_indices_and_certain = np.concatenate(
        (example_evid_indices, example_query_indices[certain_indices]),
        axis=0,
    )
    example_evid_indices_and_certain = np.sort(example_evid_indices_and_certain)

    thresholded_data = np.where(example_output > params["threshold"], 1.0, 0.0)

    _, best_assignment = run_branch_and_bound(
        idx=i,
        data_point=thresholded_data,
        evid_vars_idx=example_evid_indices_and_certain,
        temp_folder=params["temp_folder"],
        problem_uai_file=params["problem_uai_file"],
        i_bound=i_bound,
        program_path=params["program_path"],
    )

    assert (
        best_assignment[example_evid_indices] == example_output[example_evid_indices]
    ).all(), "Evidence values returned by branch and bound are not correct"

    return best_assignment


class HighUncertaintyBranchBound:

    def __init__(
        self,
        problem_uai_file,
        program_path,
        scoring_function,
        num_query_variables,
        branch_bound_max_vars,
        device,
        use_max_half_vars_branch_bound,
        threshold=0.5,
        num_processes=5,
        i_bound=10,
    ):
        self.problem_uai_file = problem_uai_file
        self.program_path = program_path
        self.scoring_function = scoring_function
        self.num_query_variables = num_query_variables
        self.branch_bound_max_vars = branch_bound_max_vars
        self.i_bound = i_bound
        # make branch bound max vars is < 1/4 * num_query_variables
        if use_max_half_vars_branch_bound:
            self.branch_bound_max_vars = min(
                self.branch_bound_max_vars, num_query_variables // 4
            )
        self.threshold = threshold
        self.device = device
        self.temp_folder = tempfile.TemporaryDirectory()
        self.num_processes = num_processes

    def select_branch_and_bound(self, all_outputs_for_pgm, query_bool, evid_bool):
        num_examples, num_vars = all_outputs_for_pgm.shape
        query_indices = np.where(query_bool)[1].reshape(num_examples, -1)
        evid_indices = np.where(evid_bool)[1].reshape(num_examples, -1)
        num_query_variables = query_indices.shape[1]
        # Create a dictionary of parameters that don't change for each example
        params = {
            "branch_bound_max_vars": self.branch_bound_max_vars,
            "threshold": self.threshold,
            "temp_folder": self.temp_folder,
            "problem_uai_file": self.problem_uai_file,
            "program_path": self.program_path,
            "num_query_variables": num_query_variables,
            "i_bound": self.i_bound,
        }

        # Create a partial function with fixed arguments
        process_example_partial = partial(
            process_example,
            all_outputs=all_outputs_for_pgm,
            query_idx=query_indices,
            evid_idx=evid_indices,
            params=params,
        )

        with tqdm(total=num_examples, desc="Processing examples") as pbar:
            best_assignments = Parallel(n_jobs=self.num_processes)(
                delayed(process_example_partial)(i)
                for i in tqdm(range(num_examples), leave=False)
            )
        best_assignments = np.array([assignment for assignment in best_assignments])
        thresholded_data = np.where(all_outputs_for_pgm > self.threshold, 1.0, 0.0)

        best_assignments_tensor = (
            torch.from_numpy(best_assignments).to(torch.double).to(self.device)
        )
        thresholded_data_tensor = (
            torch.from_numpy(thresholded_data).to(torch.double).to(self.device)
        )

        best_assignment_scores = self.scoring_function(best_assignments_tensor)
        thresholded_scores = self.scoring_function(thresholded_data_tensor)

        final_assignments = torch.where(
            best_assignment_scores[:, np.newaxis] > thresholded_scores[:, np.newaxis],
            best_assignments_tensor,
            thresholded_data_tensor,
        )

        num_examples_better_than_threshold = torch.sum(
            best_assignment_scores > thresholded_scores
        ).item()

        print(
            f"Number of examples better than threshold: {num_examples_better_than_threshold}/{num_examples}"
        )
        return final_assignments
