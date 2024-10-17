import os

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from loguru import logger
from natsort import natsorted
from tqdm import tqdm

from ..bn_class_log import BinaryBNModel
from ..mn_class_log import BinaryMNModel


def get_pgm_loss(cfg, device):
    torch_pgm_initial = None
    torch_pgm = None
    if cfg.pgm == "bn":
        cfg.pgm_model_path = os.path.join(
            cfg.pgm_model_directory, cfg.dataset, f"{cfg.dataset}.uai"
        )
        torch_pgm = BinaryBNModel(
            cfg.pgm_model_path,
            device=device,
        )
        torch_pgm_initial = torch_pgm
    elif cfg.pgm in ["mn_pairwise", "mn_higher_order"]:
        model_dir = os.path.join(
            cfg.pgm_model_directory, "downloaded_models", cfg.dataset
        )
        cfg.pgm_model_path = os.path.join(
            model_dir,
            f"{cfg.dataset}.uai",
        )
        torch_pgm_initial = BinaryMNModel(
            cfg.pgm_model_path,
            device=device,
        )
        no_zeroes_path = os.path.join(
            model_dir, "no_zeroes", f"{cfg.dataset}_no_zeroes.uai"
        )

        if os.path.exists(no_zeroes_path):
            cfg.pgm_no_zeroes_model_path = no_zeroes_path
        else:
            cfg.pgm_no_zeroes_model_path = cfg.pgm_model_path

        if cfg.use_pgm_with_fewer_cliques and cfg.embedding_type in ["gnn", "hgnn"]:
            cfg.pgm_model_path = os.path.join(
                model_dir,
                "same_sized_cliques_models",
            )
            # find the uai file in this directory
            try:
                uai_files = [
                    f for f in os.listdir(cfg.pgm_model_path) if f.endswith(".uai")
                ]
                # take the file with the largest max clique size
                uai_files_sorted = natsorted(uai_files)
                cfg.pgm_model_path = os.path.join(
                    cfg.pgm_model_path, uai_files_sorted[-1]
                )
            except Exception as e:
                cfg.pgm_model_path = os.path.join(
                    model_dir,
                    f"{cfg.dataset}.uai",
                )
        torch_pgm = BinaryMNModel(
            cfg.pgm_model_path,
            device=device,
        )

    else:
        raise NotImplementedError(f"{cfg.pgm} not implemented, use bn or mn")
    logger.info(f"Loaded PGM model from {cfg.pgm_model_path}")
    # don't allow gradient updates for the PGM
    for param in torch_pgm.parameters():
        param.requires_grad = False
    cfg.max_clique_size = torch_pgm.pgm.max_vars_in_clique
    return torch_pgm, torch_pgm_initial


def get_num_features(cfg, train_data, torch_pgm):
    num_data_features = train_data.shape[1]
    num_pgm_feature = 0
    return num_data_features, num_pgm_feature


def select_lr_scheduler(
    cfg, scheduler_name, optimizer, train_loader=None, *args, **kwargs
):
    """
    Selects and returns a learning rate scheduler from PyTorch's available lr_scheduler.

    cfg:
        scheduler_name (str): Name of the scheduler to create.
        optimizer (torch.optim.Optimizer): Optimizer for the scheduler.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torch.optim.lr_scheduler: The instantiated learning rate scheduler.

    Raises:
        ValueError: If the scheduler name is not recognized.
    """
    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.epochs // 5, gamma=0.8
        )

    elif scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, verbose=True, factor=0.8
        )
    elif scheduler_name == "OneCycleLR":
        steps_per_epoch = len(train_loader)  # Number of batches in one epoch
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            anneal_strategy="cos",  # Can be 'cos' for cosine annealing
            div_factor=25,  # Factor to divide max_lr to get the lower boundary of the learning rate
            final_div_factor=1e4,  # Factor to reduce the learning rate at the end of the cycle
            verbose=True,
        )
    elif scheduler_name == "MultiStepLR":
        return lr_scheduler.MultiStepLR(optimizer, *args, **kwargs)

    elif scheduler_name == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, *args, **kwargs)

    elif scheduler_name == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6, *args, **kwargs
        )

    elif scheduler_name == "CyclicLR":
        return lr_scheduler.CyclicLR(optimizer, *args, **kwargs)

    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, *args, **kwargs)

    elif scheduler_name == "None":
        return None
    else:
        raise ValueError(f"Unrecognized scheduler name: {scheduler_name}")
