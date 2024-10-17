import json
import os
import subprocess
from datetime import datetime

import numpy as np
from loguru import logger


def get_git_commit_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception as e:
        logger.error(f"Error obtaining Git commit hash: {e}")
        return "unknown"


def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise


def create_experiment_subdirectories(main_dir):
    subdirs = ["models", "outputs"]
    paths = {}
    for subdir in subdirs:
        dir_path = os.path.join(main_dir, subdir)
        paths[subdir] = create_directory(dir_path)
    return paths


from pathlib import Path


def prepare_experiment_config(cfg):
    config = {
        "base_dir": Path("debug" if cfg.debug else cfg.experiment_dir),
        "task_model": None,
        "extra": None,
        "debug_tuning": None,
        "use_saved_buckets": None,
        "training_mode": None,
        "params": None,
    }

    if cfg.task and cfg.model:
        config["task_model"] = (
            f"Task-{cfg.task}_PGM-{cfg.pgm}_Model-{cfg.model}_Embedding-{cfg.embedding_type}"
        )

    if cfg.debug_tuning:
        tuning_detail = "trainontest" if cfg.train_on_test_set else "standardtuning"
        config["debug_tuning"] = (
            f"{cfg.model_type}_{cfg.model_layers}_{cfg.lr_scheduler}_{cfg.train_optimizer}_{tuning_detail}"
        )

    if cfg.use_saved_buckets:
        bucket_type = "withpenalty" if cfg.add_entropy_loss else "nopenalty"
        config["use_saved_buckets"] = f"savedbuckets{bucket_type}"

    if not cfg.debug_tuning and not cfg.use_saved_buckets:
        params = [
            f"lr-{cfg.train_lr}",
            f"ep-{cfg.epochs}",
            f"bs-{cfg.batch_size}",
            f"lyrs-{cfg.model_layers}",
            f"act-{cfg.activation_function}",
            f"hact-{cfg.hidden_activation_function}",
            f"opt-{cfg.train_optimizer}-{cfg.test_optimizer}",
            f"lrs-{cfg.lr_scheduler}",
            f"emb-{cfg.embedding_type}",
            f"wd-{cfg.train_weight_decay}",
            f"dpot-{str(not cfg.no_dropout)[0]}",
            f"bn-{str(not cfg.no_batchnorm)[0]}",
            f"gclip-{str(cfg.add_gradient_clipping)[0]}",
            f"fewclq-{str(not cfg.use_pgm_with_fewer_cliques)[0]}",
        ]

        # Add embedding-specific parameters
        if cfg.embedding_type in ["hgnn", "gnn"]:
            params.extend(
                [
                    f"edim-{cfg.encoder_embedding_dim}",
                    f"gpool-{'T' if cfg.take_global_mean_pool_embedding else 'F'}",
                ]
            )
        elif cfg.embedding_type == "discrete":
            params.append(f"ns-{cfg.num_states}")

        # Add model-specific parameters
        if cfg.model == "transformer":
            params.extend(
                [f"tl-{cfg.transformer_layers}", f"pe-{cfg.positional_encoding}"]
            )

        # Add training-specific parameters
        if cfg.only_test_train_on_test_set:
            params.append("onlyTestTrue")
        if cfg.train_on_test_set:
            params.extend(
                [
                    "tm-TTT",
                    f"numItertot-{cfg.num_iter_train_on_test}",
                    f"initTot-{cfg.num_init_train_on_test}",
                ]
            )
            if cfg.use_batch_train_on_test:
                params.extend(["useBatchTot", f"bstot-{cfg.test_batch_size}"])
            if cfg.duplicate_example_train_on_test:
                params.append("dupExTot")
            if cfg.perturb_model_train_on_test:
                params.append("perturbTot")

        # Add loss-specific parameters
        if cfg.add_entropy_loss:
            params.append(f"ent-{cfg.entropy_lambda}")
        if cfg.add_distance_loss_evid_ll:
            params.append(f"distLossLL-{cfg.add_distance_loss_evid_ll}")
        if cfg.same_bucket_iter:
            params.append(f"sameBucket-{cfg.same_bucket_iter}")

        config["params"] = "_".join(filter(None, params))

        # HGNN and discrete embedding-specific parameters
        if cfg.embedding_type == "hgnn":
            second_params = [
                f"init_emb-{cfg.hgnn_initial_embedding_type}",
                f"thresh-{cfg.threshold_type}",
                f"attn_mode-{cfg.hgnn_attention_mode}",
                f"heads-{cfg.hgnn_heads}",
                f"resid-{str(cfg.encoder_residual_connections)[0]}",
                f"static-{str(cfg.use_static_node_features)[0]}",
                f"gmeanpool-{str(cfg.take_global_mean_pool_embedding)[0]}",
            ]
            config["second_params"] = "_".join(filter(None, second_params))
        elif cfg.embedding_type == "discrete":
            config["second_params"] = f"ns-{cfg.num_states}"

    # Construct the directory structure
    dir_components = [
        config["task_model"],
        config["debug_tuning"] or config["use_saved_buckets"] or "standard",
        config["params"],
        config["second_params"],
    ]
    dir_components = [comp for comp in dir_components if comp]

    config["dir_name"] = Path(*dir_components)
    config["full_path"] = config["base_dir"] / config["dir_name"]

    # Create the directory
    config["full_path"].mkdir(parents=True, exist_ok=True)

    return config


def save_experiment_metadata(output_dir, cfg):
    metadata = vars(cfg)
    metadata["git_commit_hash"] = get_git_commit_hash()
    metadata["experiment_start_time"] = datetime.now().isoformat()
    # Add system and environment information if needed
    metadata_path = os.path.join(output_dir, "metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def get_output_dir(cfg, project_name):
    config = prepare_experiment_config(cfg)

    main_dir = os.path.join(config["base_dir"], project_name, config["dir_name"])
    main_dir_path = create_directory(main_dir)

    subdirectories = create_experiment_subdirectories(main_dir_path)

    save_experiment_metadata(main_dir_path, cfg)

    return main_dir_path, subdirectories


def init_directories(cfg, project_name):
    main_dir_path, subdirectories = get_output_dir(cfg, project_name)
    model_dir = subdirectories["models"]
    model_outputs_dir = subdirectories["outputs"]
    logger.info(f"Output directory: {model_dir}")
    if not os.path.exists(model_dir) and not cfg.no_train:
        os.makedirs(model_dir)
        print("Folder created successfully!")
    else:
        print("Folder already exists!")
    cfg.model_dir = model_dir
    cfg.model_outputs_dir = model_outputs_dir


def init_directories_when_model_path_is_provided(
    cfg,
):
    prepare_directories_for_binarization(cfg)
    binarization_saved_model_path = cfg.binarization_saved_model_path
    binarization_saved_output_path = cfg.binarization_saved_output_path

    # load the outputs and store the NN LL score in cfg
    try:
        outputs = np.load(binarization_saved_output_path, allow_pickle=True)
        mean_ll_nn = np.round(outputs["mean_ll_nn"], 3).item()
        if cfg.load_saved_outputs_for_binarization:
            logger.info(
                f"Loading saved outputs for binarization from path: {binarization_saved_output_path}"
            )
            all_outputs_for_pgm = outputs["all_outputs_for_pgm"]
            all_unprocessed_data = outputs["all_unprocessed_data"]
            all_buckets = outputs["all_buckets"]
            all_nn_outputs = outputs["all_nn_outputs"]
            all_outputs_for_pgm_dict = {
                "all_unprocessed_data": all_unprocessed_data,
                "all_buckets": all_buckets,
                "all_nn_outputs": all_nn_outputs,
                "all_outputs_for_pgm": all_outputs_for_pgm,
            }
        else:
            all_outputs_for_pgm_dict = None
        cfg.prev_threshold_nn_ll = mean_ll_nn
    except Exception as e:
        logger.error(
            f"Could not load the outputs from {binarization_saved_output_path}"
        )

    assert (
        cfg.load_saved_model_for_binarization
    ), "load_saved_model_for_binarization must be True"
    model_outputs_dir = os.path.join(
        os.path.dirname(os.path.dirname(binarization_saved_output_path)),
        f"outputs_thres_type_{cfg.threshold_type}",
    )
    if cfg.threshold_type == "high_uncertainty":
        model_outputs_dir = f"{model_outputs_dir}_maxvars_{cfg.uncertainity_max_vars}"
    elif cfg.threshold_type == "branch_and_bound":
        model_outputs_dir = f"{model_outputs_dir}_bbmaxvars_{cfg.uncertainity_branch_bound_max_vars}_maxhalfvars_{str(cfg.use_max_half_vars_branch_bound)[0]}"
    elif cfg.threshold_type == "knearest_binary_vectors":
        model_outputs_dir = f"{model_outputs_dir}_k_{cfg.k_nearest_k}"
    elif cfg.threshold_type == "knearest_binary_vectors,branch_and_bound":
        model_outputs_dir = f"{model_outputs_dir}_k_{cfg.k_nearest_k}_bb_{cfg.uncertainity_branch_bound_max_vars}_maxhalfvars_{str(cfg.use_max_half_vars_branch_bound)[0]}"
    logger.add(
        os.path.join(model_outputs_dir, "logs.log"), format="{time} {level} {message}"
    )
    cfg.model_dir = os.path.dirname(binarization_saved_model_path)
    cfg.model_outputs_dir = model_outputs_dir
    return all_outputs_for_pgm_dict


def prepare_directories_for_binarization(cfg):
    if cfg.load_saved_model_for_binarization:
        assert (
            cfg.binarization_saved_model_path != ""
            or cfg.binarization_saved_output_path != ""
        ), "Please provide the model directory for a trained NN model for binarization, load_saved_model_for_binarization is True"
        assert str.endswith(cfg.binarization_saved_output_path, ".npz") or str.endswith(
            cfg.binarization_saved_model_path, ".pt"
        ), "binarization_saved_output_path must end with .npz or binarization_saved_model_path must end with models/model.pt"
        # get config
        if cfg.binarization_saved_model_path == "":
            cfg.binarization_saved_model_path = os.path.join(
                os.path.dirname(os.path.dirname(cfg.binarization_saved_output_path)),
                "models",
                "model.pt",
            )
            assert os.path.exists(
                cfg.binarization_saved_model_path
            ), "Model path does not exist"
        if cfg.binarization_saved_output_path == "":
            cfg.binarization_saved_output_path = os.path.join(
                os.path.dirname(os.path.dirname(cfg.binarization_saved_model_path)),
                "outputs",
                "test_outputs.npz",
            )
        logger.info(f"Binarization model path: {cfg.binarization_saved_model_path}")
        logger.info(
            f"Since you provided model path, we will not train the model, setting no_train to True"
        )
