import numpy as np
import torch
from loguru import logger

from lib_mpe.project_utils.data_utils import (
    get_dataloaders,
    get_mpe_solutions,
    get_num_var_in_buckets,
    init_train_test_data,
    load_data,
)
from lib_mpe.project_utils.experiment_utils import check_previous_runs, test_assertions
from lib_mpe.project_utils.model_utils import get_num_features, get_pgm_loss


class DataManager:
    def __init__(self, cfg, fabric, use_cuda=True, all_outputs_for_pgm=None):
        """
        Initializes the DataManager class.

        Args:
            cfg (ConfigManager): Configuration manager.
            use_cuda (bool): Whether to use CUDA.
        """
        self.cfg = cfg
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.test_buckets = None
        self.val_buckets = None
        self.extra_data = None
        self.num_var_in_graph = None
        self.mpe_solutions = None
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.unique_buckets = None
        self.load_data()
        self.get_pgm_loss()
        self.init_train_test_data()
        self.best_model_info = self.check_previous_runs(all_outputs_for_pgm)
        self.add_extra_data()
        logger.info(f"Train data shape: {self.train_data.shape}")
        self.mpe_solutions = self.get_mpe_solutions()

        self.init_dataloaders(
            fabric, self.torch_pgm, self.num_variables_in_buckets, use_cuda
        )

    def load_data(self):
        """
        Loads the data from the given path.
        """
        data, self.extra_data, buckets = load_data(self.cfg, self.cfg.dataset)
        self.train_data, self.test_data, self.val_data = (
            data["train"],
            data["test"],
            data["valid"],
        )
        self.val_buckets, self.test_buckets, self.train_buckets = (
            buckets["valid"],
            buckets["test"],
            buckets["train"],
        )

    def init_train_test_data(self):
        """
        Initializes the train, test, and validation data.
        """
        (
            self.test_data,
            self.test_buckets,
            self.train_data,
            self.val_data,
            self.val_buckets,
        ) = init_train_test_data(
            self.cfg,
            self.test_data,
            self.test_buckets,
            self.val_data,
            self.val_buckets,
            self.train_data,
        )
        self.num_var_in_graph = self.train_data.shape[1]

    def get_mpe_solutions(self):
        """
        Gets the MPE solutions.
        """
        mpe_solutions = get_mpe_solutions(self.cfg)
        if mpe_solutions is None:
            logger.info("No MPE solutions found. The baseline did not finish running")
            str = input("Press Enter to continue...")
            # if user presses enter, we will continue
            if str == "":
                mpe_solutions = {
                    "test_mpe_output": np.zeros_like(self.test_data),
                    "test_root_ll_pgm": np.zeros(self.test_data.shape[0]),
                }
            else:
                raise ValueError("Please press enter to continue")
        return mpe_solutions

    def check_previous_runs(self, all_outputs_for_pgm):
        """
        Checks for previous runs.
        """

        best_model_info, self.train_data, self.test_data, self.val_data = (
            check_previous_runs(
                self.cfg,
                self.train_data,
                self.test_data,
                self.val_data,
                all_outputs_for_pgm,
            )
        )
        return best_model_info

    def add_extra_data(self):
        """
        Adds extra data to the training data.
        """
        if not self.cfg.no_extra_data:
            self.train_data = torch.cat((self.train_data, self.extra_data), dim=0)
            logger.info("We are adding extra sampled data")

    def init_dataloaders(self, fabric, torch_pgm, num_variables_in_buckets, use_cuda):
        """
        Initializes the dataloaders.

        Args:
            torch_pgm (bool): PGM implemented in PyTorch.
            num_variables_in_buckets (int): Number of variables in the buckets.
            use_cuda (bool): Whether to use CUDA.
        """
        self.unique_buckets = None
        self.train_loader, self.test_loader, self.val_loader = get_dataloaders(
            self.cfg,
            self.train_data,
            self.test_data,
            self.val_data,
            self.train_buckets,
            self.val_buckets,
            self.test_buckets,
            use_cuda,
            self.unique_buckets,
            torch_pgm,
            num_variables_in_buckets,
        )
        if self.cfg.use_static_node_features:
            # add 5 to the initialize_embed_size because we are adding 5 static features: degree, clustering, betweenness, closeness, eigenvector
            self.cfg.initialize_embed_size += 5
        self.setup_fabric_dataloaders(fabric)

    def setup_fabric_dataloaders(self, fabric):
        """
        Sets up the dataloaders for the fabric.

        Args:
            fabric (Fabric): Fabric object.
        """
        self.train_loader = fabric.setup_dataloaders(self.train_loader)
        self.test_loader = fabric.setup_dataloaders(self.test_loader)
        self.val_loader = fabric.setup_dataloaders(self.val_loader)

    def get_pgm_loss(self):
        """
        Gets the PGM loss.
        """
        self.torch_pgm, self.torch_pgm_initial = get_pgm_loss(self.cfg, self.cfg.device)
        self.num_data_features, self.num_pgm_feature = get_num_features(
            self.cfg, self.train_data, self.torch_pgm
        )
        self.num_variables_in_buckets = get_num_var_in_buckets(
            self.cfg, self.train_data.shape[1]
        )
        self.num_query_variables = self.num_variables_in_buckets[1]
        self.num_outputs = self.train_data.shape[1]

    def return_all_info(self):
        """
        Returns the information about the data.
        """
        return (
            self.train_data,
            self.test_data,
            self.val_data,
            self.test_buckets,
            self.val_buckets,
            self.num_var_in_graph,
            self.mpe_solutions,
            self.train_loader,
            self.test_loader,
            self.val_loader,
            self.unique_buckets,
            self.num_data_features,
            self.num_pgm_feature,
            self.num_variables_in_buckets,
            self.torch_pgm,
            self.num_query_variables,
            self.num_outputs,
        )

    def return_info(self):
        """
        Returns the information about the data.
        """
        return (
            self.train_loader,
            self.num_data_features,
            self.num_pgm_feature,
            self.num_query_variables,
            self.num_outputs,
        )
