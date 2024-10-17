import os

import lightning as L
import numpy as np
import torch
import yaml
from loguru import logger

from lib_mpe.create_output_dir import (
    init_directories,
    init_directories_when_model_path_is_provided,
)
from lib_mpe.project_utils.experiment_utils import infer_cfg_from_saved_outputs
from lib_mpe.project_utils.logging_utils import init_logger_and_wandb


class ConfigManager:
    def __init__(self, cfg, project_name):
        """
        Initializes the ConfigManager class.

        Args:
            cfg (dict): Configuration dictionary.
            project_name (str): Name of the project.
        """
        self.cfg = cfg
        self.device = None
        self.use_cuda = None
        self.use_mps = None
        self.init_config(project_name)
        self.cfg.device = self.device

    def return_info(self):
        """
        Returns the configuration, device, CUDA usage, and MPS usage.

        Returns:
            tuple: A tuple containing the configuration, device, CUDA usage, and MPS usage.
        """
        return self.cfg, self.device, self.use_cuda, self.use_mps

    def init_config(self, project_name):
        """
        Initializes the configuration, seed, and directories.

        Args:
            project_name (str): Name of the project.
        """
        if self.cfg.seed:
            torch.manual_seed(self.cfg.seed)
            L.seed_everything(self.cfg.seed, workers=True)
        if (
            self.cfg.load_saved_model_for_binarization
            or self.cfg.load_saved_outputs_for_binarization
        ):
            assert (
                self.cfg.infer_cfg_from_saved_outputs
            ), "Must set infer_cfg_from_saved_outputs to True when loading saved model for binarization: otherwise dataset details will not be consistent"
            # We are going to perform a different binarization for already trained models
            logger.info(
                "Loading saved model for binarization from path: {}".format(
                    self.cfg.binarization_saved_output_path
                )
            )
            self.all_outputs_for_pgm_dict = (
                init_directories_when_model_path_is_provided(
                    self.cfg,
                )
            )
            if self.cfg.infer_cfg_from_saved_outputs:
                # infer the cfg from the saved outputs
                logger.info("Inferring cfg from saved outputs")
                logger.info(
                    f"Note that this should only be used when cfg.load_saved_model_for_binarization is true"
                )
                self.cfg = infer_cfg_from_saved_outputs(self.cfg)
        else:
            # need to train the model
            logger.info("Creating output directories")
            init_directories(self.cfg, project_name)
        self.device, self.use_cuda, self.use_mps = init_logger_and_wandb(
            project_name, self.cfg
        )
        torch.manual_seed(self.cfg.seed)
        # save the config to the yaml file in cfg.model_outputs_dir
        with open(os.path.join(self.cfg.model_outputs_dir, "config.yaml"), "w") as f:
            yaml.dump(self.cfg, f)
        logger.info(
            f"Config saved to {os.path.join(self.cfg.model_outputs_dir, 'config.yaml')}"
        )
