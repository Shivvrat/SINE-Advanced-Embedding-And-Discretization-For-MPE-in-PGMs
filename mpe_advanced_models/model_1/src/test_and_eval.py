from __future__ import print_function

import math
import os
import time

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from nn_scripts import train_and_validate_single_example, validate

import wandb
from lib_mpe.inference_blocks.high_uncert_branch_bound import HighUncertaintyBranchBound
from lib_mpe.inference_blocks.high_uncert_thres import (
    HighUncertaintyVariableBasedThresholder,
)
from lib_mpe.inference_blocks.kn_binary_vectors import KNearestBinaryVectorsFinder
from lib_mpe.project_utils.logging_utils import save_args_as_yaml
from lib_mpe.project_utils.results import (
    construct_data_row,
    construct_header,
    write_to_csv,
)


def test_and_process_outputs(
    cfg,
    device,
    fabric,
    pgm,
    pgm_initial,
    model_dir,
    model_outputs_dir,
    train_loader,
    test_loader,
    val_loader,
    mpe_solutions,
    model,
    optimizer,
    best_loss,
    counter,
    all_train_losses,
    all_val_losses,
    best_model_info,
    num_data_features,
    num_pgm_feature,
    num_outputs,
    num_query_variables,
):
    # Load the best model for further use or testing
    logger.info("Loading best model...")
    model.load_state_dict(best_model_info["model_state"])
    os.makedirs(model_outputs_dir, exist_ok=True)
    if not cfg.debug_tuning and not cfg.no_train and cfg.evaluate_training_set:
        dataset_types = ["train", "val", "test"]
    else:
        dataset_types = ["test"]
    ll_scores_nn = {data_type: None for data_type in dataset_types}
    ll_scores_pgm = {data_type: None for data_type in dataset_types}
    save_args_as_yaml(cfg, os.path.join(model_outputs_dir, "cfg.yaml"))

    for data_type in dataset_types:
        if data_type == "train":
            loader = train_loader
        elif data_type == "test":
            loader = test_loader
        elif data_type == "val":
            loader = val_loader

        if not cfg.train_on_test_set or (data_type in ["train", "val"]):
            validation_function = validate
            (
                _,
                _,
                _,
                all_unprocessed_data,
                all_nn_outputs,
                all_outputs_for_pgm,
                all_buckets,
            ) = validation_function(cfg, model, pgm, device, loader, best_loss, counter)
        else:
            validation_function = train_and_validate_single_example

            (
                _,
                _,
                _,
                all_unprocessed_data,
                all_nn_outputs,
                all_outputs_for_pgm,
                all_buckets,
            ) = validation_function(
                cfg,
                model,
                pgm,
                fabric,
                loader,
                optimizer,
                best_loss,
                counter,
                device,
                num_data_features,
                num_pgm_feature,
                num_outputs,
                num_query_variables,
            )
            # logger.info(f"Outputs of the model - {all_outputs_for_pgm}")
        if mpe_solutions is not None:
            mpe_output = mpe_solutions[f"{data_type}_mpe_output"]
            root_ll_pgm = mpe_solutions[f"{data_type}_root_ll_pgm"]
        else:
            mpe_output = None
            root_ll_pgm = None
        root_ll_nn_torch, root_ll_pgm, mpe_output, thresholded_output = get_ll_scores(
            cfg,
            pgm_initial,
            all_unprocessed_data,
            all_outputs_for_pgm,
            all_buckets,
            num_query_variables,
            mpe_output,
            root_ll_pgm,
            device,
        )
        mean_ll_pgm = np.mean(root_ll_pgm)
        std_ll_pgm = np.std(root_ll_pgm)
        mean_ll_nn = np.mean(root_ll_nn_torch)
        std_ll_nn = np.std(root_ll_nn_torch)
        logger.info(f"Root LL NN Ours {data_type}: {mean_ll_nn}")
        logger.info(f"Root LL PGM {data_type}: {mean_ll_pgm}")
        ll_scores_nn[data_type] = mean_ll_nn
        ll_scores_pgm[data_type] = mean_ll_pgm

        wandb.log(
            {
                f"Root LL NN {data_type}": mean_ll_nn,
                f"Root LL PGM {data_type}": mean_ll_pgm,
            }
        )
        os.makedirs(model_outputs_dir, exist_ok=True)
        output_path = f"{model_outputs_dir}/{data_type}_outputs.npz"
        np.savez(
            output_path,
            all_unprocessed_data=all_unprocessed_data,
            all_outputs_for_pgm=all_outputs_for_pgm,
            all_buckets=all_buckets,
            all_nn_outputs=all_nn_outputs,
            mpe_output=mpe_output,
            root_ll_nn_torch=root_ll_nn_torch,
            root_ll_pgm=root_ll_pgm,
            mean_ll_nn=mean_ll_nn,
            mean_ll_nn_library=mean_ll_nn,
            mean_ll_pgm=mean_ll_pgm,
            std_ll_nn=std_ll_nn,
            std_ll_pgm=std_ll_pgm,
            runtime=time.time() - wandb.run.start_time,
            thresholded_output=thresholded_output,
        )
        # Save the metrics to a common file for models and datasets
        if data_type == "test":
            metrics = {
                "mean_ll_nn": mean_ll_nn,
                "mean_ll_pgm": mean_ll_pgm,
                "std_ll_nn": std_ll_nn,
                "std_ll_pgm": std_ll_pgm,
            }
            if cfg.prev_threshold_nn_ll != 0.0:
                metrics["prev_threshold_nn_ll"] = cfg.prev_threshold_nn_ll
            model_name = cfg.model
            dataset_name = cfg.dataset
            # get wandb run id
            run_id = wandb.run.id
            outputs_path = f"{model_outputs_dir}/{data_type}_outputs.npz"
            runtime = time.time() - wandb.run.start_time
            # Save all these detils in a csv file
            common_results_path = os.path.join(
                "common_results", dataset_name, model_name
            )
            os.makedirs(common_results_path, exist_ok=True)
            csv_path = os.path.join(common_results_path, "results.csv")
            header = construct_header(cfg)
            data_row = construct_data_row(run_id, metrics, outputs_path, runtime, cfg)
            write_to_csv(csv_path, header, data_row)

        cfg.outputs_path = f"{model_outputs_dir}/{data_type}_outputs.npz"

    alert_str = ""
    for data_type in ll_scores_nn:
        alert_str += f"{model_outputs_dir} \n Root LL NN ({data_type.capitalize()}): {ll_scores_nn[data_type]}, Root LL PGM ({data_type.capitalize()}): {ll_scores_pgm[data_type]}, \n"
    if cfg.prev_threshold_nn_ll != 0.0:
        alert_str += f"Previous threshold NN LL (TEST): {cfg.prev_threshold_nn_ll}\n"
    wandb.alert(
        title=f"Dataset: {dataset_name}, {cfg.task}",
        text=alert_str,
    )
    logger.info(alert_str)
    logger.info(f"Model saved at {model_dir}")
    logger.info(f"Model outputs saved at {model_outputs_dir}")


def threshold_array(arr, threshold, value_less, value_more):
    """
    The function `threshold_array` takes an array `arr`, a threshold value, and two values `value_less`
    and `value_more`, and returns a new array where values less than or equal to the threshold are
    replaced with `value_less` and values greater than the threshold are replaced with `value_more`.

    :param arr: The input array on which the threshold operation will be performed
    :param threshold: The threshold is a value that determines whether elements in the array are
    considered less than or equal to the threshold or greater than the threshold
    :param value_less: The value to assign to elements in the array that are less than or equal to the
    threshold
    :param value_more: The value to assign to elements in the array that are greater than the threshold
    :return: a new array where the elements that are less than or equal to the threshold are replaced
    with the value_less, and the elements that are greater than the threshold are replaced with the
    value_more.
    """
    if type(arr) == torch.Tensor:
        return torch.where(arr <= threshold, value_less, value_more)
    else:
        return np.where(arr <= threshold, value_less, value_more)


def remove_nan_rows(arr):
    """
    The function removes rows containing NaN values from a given array.

    :param arr: The parameter "arr" is expected to be a numpy array
    :return: an array with the rows that do not contain any NaN values.
    """
    return arr[~np.isnan(arr).any(axis=1)]


@torch.no_grad()
def evaluate_nn(
    cfg, pytorch_pgm, all_outputs_for_pgm, num_query_variables, query_bool, evid_bool
):
    all_outputs_for_pgm = torch.tensor(
        all_outputs_for_pgm, dtype=torch.float64, device=cfg.device
    )
    logger.info(f"Threshold type: {cfg.threshold_type}")

    threshold_methods = {
        "basic": lambda: threshold_array(all_outputs_for_pgm, cfg.threshold, 0, 1),
        "high_uncertainty": lambda: HighUncertaintyVariableBasedThresholder(
            cfg.uncertainity_max_vars,
            pytorch_pgm.evaluate,
            num_query_variables,
            cfg.device,
            cfg.threshold,
        ).select_high_uncertainty(all_outputs_for_pgm, query_bool),
        "branch_and_bound": lambda: HighUncertaintyBranchBound(
            cfg.pgm_no_zeroes_model_path,
            cfg.branch_bound_program_path,
            pytorch_pgm.evaluate,
            num_query_variables,
            cfg.uncertainity_branch_bound_max_vars,
            cfg.device,
            cfg.use_max_half_vars_branch_bound,
        ).select_branch_and_bound(
            all_outputs_for_pgm.cpu().numpy(), query_bool, evid_bool
        ),
        "knearest_binary_vectors": lambda: KNearestBinaryVectorsFinder(
            cfg.k_nearest_k, pytorch_pgm.evaluate, num_query_variables, cfg.device
        ).select_k_nearest(all_outputs_for_pgm, query_bool),
    }

    if cfg.threshold_type in threshold_methods:
        all_outputs_for_pgm = threshold_methods[cfg.threshold_type]()
    elif cfg.threshold_type == "knearest_binary_vectors,branch_and_bound":
        kn_outputs = threshold_methods["knearest_binary_vectors"]()
        bnb_outputs = threshold_methods["branch_and_bound"]()
        kn_scores = pytorch_pgm.evaluate(kn_outputs)
        bnb_scores = pytorch_pgm.evaluate(bnb_outputs)
        all_outputs_for_pgm = torch.where(
            kn_scores.unsqueeze(1) > bnb_scores.unsqueeze(1), kn_outputs, bnb_outputs
        )

    if isinstance(all_outputs_for_pgm, np.ndarray):
        all_outputs_for_pgm = torch.tensor(
            all_outputs_for_pgm, dtype=torch.float64, device=cfg.device
        )

    root_ll_our_pgm = pytorch_pgm.evaluate(all_outputs_for_pgm)
    thresholded_output = all_outputs_for_pgm.cpu().numpy()
    return root_ll_our_pgm.cpu().numpy(), thresholded_output


def get_ll_scores(
    cfg,
    torch_pgm,
    all_unprocessed_data,
    all_outputs_for_pgm,
    all_buckets,
    num_query_variables,
    mpe_output,
    root_ll_pgm_lib,
    device,
):
    """
    The function `get_ll_scores` calculates the log-likelihood scores for a given set of inputs and
    outputs using a root pgm (Sum-Product Network) and a Torch pgm (PyTorch implementation of pgm).

    :param cfg: The `cfg` parameter is a dictionary or object that contains various arguments or
    settings for the function. It is used to configure the behavior of the function
    :param root_pgm: The `root_pgm` parameter is a root node of a Sum-Product Network (pgm). It
    represents the top-level node of the pgm, from which all other nodes can be reached
    :param torch_pgm: The `torch_pgm` parameter is a PyTorch implementation of a Sum-Product Network
    (pgm) model. It is used to evaluate the log-likelihood of the pgm model given the input data
    :param all_unprocessed_data: A list or array containing the unprocessed data for all instances. Each
    instance should be a list or array of values for each variable
    :param all_outputs_for_pgm: A numpy array containing the outputs of the pgm for all data points. It
    has shape (num_data_points, num_variables)
    :param all_buckets: `all_buckets` is a dictionary that contains different types of variables and
    their corresponding buckets. The keys in the dictionary represent the type of variables, and the
    values are arrays that indicate which variables belong to each type. For example,
    `all_buckets["query"]` contains an array that indicates which variables
    :param device: The `device` parameter is used to specify the device (e.g., CPU or GPU) on which the
    computations should be performed. It is typically a string indicating the device, such as "cpu" or
    "cuda:0"
    :return: The function `get_ll_scores` returns four values: `root_ll_our_nn`, `root_ll_nn`,
    `root_ll_pgm`, and `mpe_output`.
    """
    # use precomputed mpe output
    mpe_output = mpe_output
    mean_ll_pgm = root_ll_pgm_lib
    if cfg.pgm in ["mn_pairwise", "mn_higher_order", "bn"]:
        mpe_output = mpe_output[: len(all_outputs_for_pgm)]
        torch_mpe_output = (
            torch.FloatTensor(mpe_output).to(cfg.device).to(torch.float64)
        )
        root_ll_our_pgm = torch_pgm.evaluate(torch_mpe_output)
        root_ll_pgm = root_ll_our_pgm.detach().cpu().numpy()
        # check if mean_ll_pgm is not zero dimensional
        if not (
            mean_ll_pgm.shape == ()
            and mean_ll_pgm.dtype == object
            and mean_ll_pgm.item() is None
        ):
            if cfg.debug:
                assert np.allclose(np.mean(root_ll_pgm), mean_ll_pgm, atol=200)
            else:
                assert np.allclose(np.mean(root_ll_pgm), mean_ll_pgm, atol=2)
        if np.all(np.isnan(root_ll_pgm)):
            root_ll_pgm = np.array([-math.inf])

    root_ll_our_nn, thresholded_output = evaluate_nn(
        cfg,
        torch_pgm,
        all_outputs_for_pgm=all_outputs_for_pgm,
        num_query_variables=num_query_variables,
        query_bool=all_buckets["query"],
        evid_bool=all_buckets["evid"],
    )
    logger.info(f"Root ll PGM our implementation {np.mean(root_ll_pgm)}")
    logger.info(f"Root ll NN {np.mean(root_ll_our_nn)}")
    mpe_output = torch_mpe_output.detach().cpu().numpy()
    # root_ll_nn = np.average(root_ll_nn)
    if cfg.debug:
        # join the ll scores for the test set
        try:
            joint_scores = np.vstack((root_ll_pgm, root_ll_our_nn))
            print(joint_scores)
        except:
            pass
    return root_ll_our_nn, root_ll_pgm, mpe_output, thresholded_output
