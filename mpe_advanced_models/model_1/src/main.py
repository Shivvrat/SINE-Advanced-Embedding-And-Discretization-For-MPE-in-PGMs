from __future__ import print_function

import lightning as L
import rootutils
import torch
from loguru import logger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
    data_manager = DataManager(cfg, fabric, config_manager.use_cuda)
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

    model_manager = ModelManager(
        cfg,
        cfg.device,
        fabric,
        num_data_features,
        num_pgm_feature,
        num_outputs,
        num_query_variables,
        train_loader,
    )
    ###########################################################################
    trainer = Trainer(cfg, fabric, data_manager, model_manager)
    if not cfg.no_train:
        trainer.train_model()

    if not cfg.not_save_model and not cfg.no_test:
        trainer.test_and_save_model()


if __name__ == "__main__":
    cfg, project_name = define_arguments()
    runner(cfg, project_name)

    # experiments are finished
