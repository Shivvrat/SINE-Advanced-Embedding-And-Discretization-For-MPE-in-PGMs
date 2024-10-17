import os

import torch
from loguru import logger

from lib_mpe.project_utils.logging_utils import load_args_from_yaml
from lib_mpe.project_utils.str_repo import debug_string


def check_previous_runs(cfg, train_data, test_data, val_data, all_outputs_for_pgm=None):
    """
    Checks for previous runs and adjusts configuration accordingly.

    If not in debug mode, it checks for existing model outputs and model files,
    and sets flags in cfg to avoid retraining or retesting if not necessary.
    In debug mode, it slices the data to smaller sizes for faster debugging.

    Returns:
        best_model_info (dict): Information about the best model.
        out_train_data: Potentially modified training data.
        out_test_data: Potentially modified testing data.
        out_val_data: Potentially modified validation data.
    """
    best_model_info = {
        "epoch": None,
        "model_state": None,
        "optimizer_state": None,
    }

    if cfg.debug:
        logger.info(debug_string)
        out_train_data = train_data[:3000]
        out_test_data = test_data[:100]
        out_val_data = val_data[:500]
    if not cfg.debug or cfg.load_saved_model_for_binarization:
        # if load_saved_model_for_binarization is true, then we want to load the model and not run training
        test_outputs_path = os.path.join(cfg.model_outputs_dir, "test_outputs.npz")
        if "model.pt" not in cfg.model_dir:
            model_path = os.path.join(cfg.model_dir, "model.pt")
        else:
            model_path = cfg.model_dir
        if cfg.replace:
            assert (
                not cfg.load_saved_model_for_binarization
            ), "Cannot replace model when load_saved_model_for_binarization is true"
            logger.info("Replacing existing model")
            cfg.no_train = False
            cfg.no_test = False
        else:
            test_outputs_exist = os.path.exists(test_outputs_path)
            model_exists = os.path.exists(model_path)
            if test_outputs_exist and not cfg.debug:
                logger.info("Model outputs already stored")
                cfg.no_test = True
            else:
                cfg.no_test = False

            if not cfg.no_train:
                if model_exists:
                    if (not cfg.no_test and not cfg.only_test_train_on_test_set) and (
                        not cfg.load_saved_outputs_for_binarization
                        or (all_outputs_for_pgm is None)
                    ):
                        # If test outputs are not stored, we need to train the model
                        logger.info(
                            "Model already trained, but test outputs missing. Preparing to test."
                        )
                        try:
                            cfg = load_args_from_yaml(
                                os.path.join(cfg.model_dir, "cfg.yaml")
                            )
                        except FileNotFoundError:
                            logger.info("Could not load cfg.yaml")
                        try:
                            best_model_info = torch.load(model_path)
                            cfg.no_train = True
                        except FileNotFoundError:
                            logger.info("Could not load model.pt")
                            cfg.no_train = False
                    else:
                        logger.info(
                            "Test outputs already stored. No training required."
                        )
                        cfg.no_train = True
                else:
                    logger.info("No existing model found. Training required.")
                    cfg.no_train = False

    if not cfg.debug:
        out_train_data = train_data
        out_test_data = test_data
        out_val_data = val_data

    if cfg.no_train and cfg.no_test:
        logger.info("No training or testing required")

    if cfg.only_test_train_on_test_set:
        assert cfg.no_train, "Cannot do only test when no train is false"

    return best_model_info, out_train_data, out_test_data, out_val_data


def infer_cfg_from_saved_outputs(cfg):
    """
    Infer the cfg from the saved outputs.
    """
    cfg_path = os.path.join(
        os.path.dirname(cfg.binarization_saved_output_path), "cfg.yaml"
    )
    loaded_cfg = load_args_from_yaml(cfg_path)

    # Update the cfg with the loaded cfg for specified arguments
    cfg.query_prob = loaded_cfg.get("query_prob", cfg.query_prob)
    cfg.evidence_prob = loaded_cfg.get("evidence_prob", cfg.evidence_prob)
    cfg.model = loaded_cfg.get("model", cfg.model)
    cfg.model_layers = loaded_cfg.get("model_layers", cfg.model_layers)
    cfg.dataset = loaded_cfg.get("dataset", cfg.dataset)
    cfg.task = loaded_cfg.get("task", cfg.task)
    cfg.partition_type = loaded_cfg.get("partition_type", cfg.partition_type)
    cfg.embedding_type = loaded_cfg.get("embedding_type", cfg.embedding_type)
    cfg.pgm = loaded_cfg.get("pgm", cfg.pgm)
    cfg.encoder_embedding_dim = loaded_cfg.get(
        "encoder_embedding_dim", cfg.encoder_embedding_dim
    )
    cfg.train_optimizer = loaded_cfg.get("train_optimizer", cfg.train_optimizer)
    logger.info("Updated cfg")
    logger.info(
        f"Updated values: \n query_prob: {cfg.query_prob}\n model: {cfg.model}\n model_layers: {cfg.model_layers}\n dataset: {cfg.dataset}\n task: {cfg.task}\n partition_type: {cfg.partition_type}\n embedding_type: {cfg.embedding_type}\n pgm: {cfg.pgm}\n encoder_embedding_dim: {cfg.encoder_embedding_dim}\n train_optimizer: {cfg.train_optimizer}"
    )
    return cfg


def test_assertions(cfg, best_model_info):
    assert (
        best_model_info["epoch"] is None
        if cfg.only_test_train_on_test_set
        else isinstance(best_model_info["epoch"], int)
    ), "We should not have a trained model when only-test-train-on-test-set is true, else num epochs should be an integer"
    assert (
        cfg.only_test_train_on_test_set == cfg.train_on_test_set
        if cfg.only_test_train_on_test_set
        else True
    ), "only-test-train-on-test-set should be true only if train-on-test-set is true"
