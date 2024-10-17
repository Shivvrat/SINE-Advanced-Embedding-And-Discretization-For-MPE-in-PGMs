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
from mpe_advanced_models.model_1.src.test_and_eval import get_ll_scores


def only_threshold_test(
    cfg,
    model_outputs_dir,
    model_dir,
    mpe_solutions,
    pgm_initial,
    all_unprocessed_data,
    all_outputs_for_pgm,
    all_buckets,
    all_nn_outputs,
    num_query_variables,
    device,
    data_type="test",
):
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
        common_results_path = os.path.join("common_results", dataset_name, model_name)
        os.makedirs(common_results_path, exist_ok=True)
        csv_path = os.path.join(common_results_path, "results.csv")
        header = construct_header(cfg)
        data_row = construct_data_row(run_id, metrics, outputs_path, runtime, cfg)
        write_to_csv(csv_path, header, data_row)

    cfg.outputs_path = f"{model_outputs_dir}/{data_type}_outputs.npz"

    alert_str = ""
    alert_str += f"{model_outputs_dir} \n Root LL NN ({data_type.capitalize()}): {mean_ll_nn}, Root LL PGM ({data_type.capitalize()}): {mean_ll_pgm}, \n"
    if cfg.prev_threshold_nn_ll != 0.0:
        alert_str += f"Previous threshold NN LL (TEST): {cfg.prev_threshold_nn_ll}\n"
    wandb.alert(
        title=f"Dataset: {dataset_name}, {cfg.task}",
        text=alert_str,
    )
    logger.info(alert_str)
    logger.info(f"Model saved at {model_dir}")
    logger.info(f"Model outputs saved at {model_outputs_dir}")
