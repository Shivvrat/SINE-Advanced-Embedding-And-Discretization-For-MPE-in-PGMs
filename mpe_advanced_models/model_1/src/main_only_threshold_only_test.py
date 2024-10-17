from __future__ import print_function

import lightning as L
import rootutils
import torch
from loguru import logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from only_threshold_test import only_threshold_test

# trainers
from trainer import Trainer

# Get classes
from util_classes.config import ConfigManager
from util_classes.data_manager import DataManager
from util_classes.model_manager import ModelManager

# load arguments
from utils_folder.arguments import define_arguments

torch.set_default_dtype(torch.float64)


@logger.catch
def runner(cfg, project_name):

    config_manager = ConfigManager(cfg, project_name)
    fabric = L.Fabric(
        accelerator=cfg.device,
        devices=1,
        precision="64",
    )
    fabric.launch()
    data_manager = DataManager(
        cfg, fabric, config_manager.use_cuda, config_manager.all_outputs_for_pgm_dict
    )
    if cfg.no_train and cfg.no_test:
        logger.info("No training or testing required. Experiment already finished.")
        return
    (
        train_loader,
        num_data_features,
        num_pgm_feature,
        num_query_variables,
        num_outputs,
    ) = data_manager.return_info()

    # update config_manager.all_outputs_for_pgm_dict["all_outputs_for_pgm"] based on config_manager.all_outputs_for_pgm_dict["all_nn_outputs"]
    nn_outputs = config_manager.all_outputs_for_pgm_dict["all_nn_outputs"]
    buckets = data_manager.test_buckets

    test_data = data_manager.test_data
    all_outputs_for_pgm = nn_outputs[: test_data.shape[0]].copy()
    all_outputs_for_pgm[buckets["evid"]] = test_data[buckets["evid"]]
    if not cfg.not_save_model and not cfg.no_test:
        only_threshold_test(
            cfg,
            cfg.model_outputs_dir,
            cfg.model_dir,
            data_manager.mpe_solutions,
            data_manager.torch_pgm_initial,
            config_manager.all_outputs_for_pgm_dict["all_unprocessed_data"],
            all_outputs_for_pgm,
            data_manager.test_buckets,
            config_manager.all_outputs_for_pgm_dict["all_nn_outputs"],
            num_query_variables,
            cfg.device,
            data_type="test",
        )


if __name__ == "__main__":
    cfg, project_name = define_arguments()
    runner(cfg, project_name)

    # experiments are finished
