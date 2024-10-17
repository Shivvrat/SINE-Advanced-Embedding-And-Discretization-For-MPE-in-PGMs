import argparse
import os
import re
import subprocess
import time
from pprint import pprint

from loguru import logger

from mpe_advanced_models.model_1.src.utils_folder.dataset_lists import (
    complex_datasets,
    pairwise_datasets,
)


def define_arguments():
    parser = argparse.ArgumentParser(description="Advanced Models for MPE Experiments")

    # Debugging and Environment Settings
    parser.add_argument(
        "--no-debug", action="store_true", default=False, help="Enable debug mode."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA training."
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="Disable macOS GPU training.",
    )
    parser.add_argument(
        "--data_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help='Select the device to use: "cpu" or "gpu".',
    )

    parser.add_argument(
        "--seed", type=int, default=27, help="Random seed (default: 1)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Quickly check a single pass.",
    )
    parser.add_argument(
        "--pgm",
        choices=["bn", "mn_pairwise", "mn_higher_order"],
        default="mn_higher_order",
        help="PGM to use (default: mn_higher_order).",
    )

    parser.add_argument(
        "--baseline",
        choices=["maxprod"],
        default="maxprod",
        help="Baseline to use (default: maxprod).",
    )
    # Model and Training Settings
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "nn",
            "transformer",
        ],
        default="nn",
        help="Model to use (default: nn).",
    )
    parser.add_argument(
        "--model-type",
        default="2",
        choices=["2"],
        help="Model type to train (default: 2).",
    )
    parser.add_argument(
        "--use-single-model",
        action="store_true",
        default=False,
        help="Use one model per dataset.",
    )
    parser.add_argument(
        "--input-type",
        default="data",
        choices=[
            "data",
        ],
        help="Model type to train (default: data).",
    )
    # input can be discrete or continuous
    parser.add_argument(
        "--embedding-type",
        default="hgnn",
        choices=["discrete", "hgnn", "gnn"],
        help="Embedding type (default: discrete). This is how we embed the three types of inputs (query, evidence, unobs). None is used when we are using a GNN.",
    )
    parser.add_argument(
        "--hgnn_initial_embedding_type",
        type=str,
        default="discrete",
        choices=["discrete", "learnable_per_node", "learnable_per_node_value"],
        help="Type of embedding to use for the initial layer of the HGCN (default: discrete).",
    )
    parser.add_argument(
        "--hypergraph-class",
        default="HypergraphConv",
        choices=[
            "hgnn",  # does not allow edge embedding
            "hnhn",  # does not allow edge embedding
            "hypergcn",  # does not allow edge embedding
            "dhcf",  # does not allow edge embedding
            "unigcn",  # does not allow edge embedding
            "unigat",  # does not allow edge embedding
            "unignn",  # does not allow edge embedding
            "unisage"  # does not allow edge embedding,
            "unigin",  # does not allow edge embedding
            "HypergraphConv",  # allows edge embedding
        ],
        help="Hypergraph class (default: hgnn).",
    )
    parser.add_argument(
        "--hgnn_attention_mode",
        type=str,
        choices=["node", "edge"],
        default="node",
        help="Attention mode in the HGCN.",
    )
    parser.add_argument(
        "--hgnn_heads",
        type=int,
        default=1,
        help="Number of heads in the HGCN.",
    )
    parser.add_argument(
        "--use-static-node-features",
        action="store_true",
        default=False,
        help="Use node features.",
    )
    parser.add_argument(
        "--encoder-embedding-dim",
        type=str,
        default="64",
        help="Embedding size for the encoder model (default: 512).",
    )
    parser.add_argument(
        "--encoder-residual-connections",
        action="store_true",
        default=False,
        help="Use residual connections in the encoder model (default: False).",
    )
    parser.add_argument(
        "--initialize_embed_size",
        type=int,
        default=2,
        help="Embedding size for the encoder model (default: 512).",
    )
    parser.add_argument(
        "--take_global_mean_pool_embedding",
        action="store_true",
        default=False,
        help="Take global mean pool embedding.",
    )
    parser.add_argument(
        "--max_clique_size",
        type=int,
        help="Max vars in clique for the PGM.",
    )
    parser.add_argument(
        "--num-states",
        type=int,
        default=4,
        help="Number of states for discrete variables (default: 4).",
    )
    parser.add_argument(
        "--evaluate-training-set",
        action="store_true",
        default=False,
        help="Evaluate on training set.",
    )

    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=1,
        help="Number of layers for encoder model (default: 1).",
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        default=3,
        help="Number of layers for mpe solver (default: 1).",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=1,
        help="Number of layers for transformer model (default: 1).",
    )
    parser.add_argument(
        "--positional-encoding",
        choices=["sinusoidal", "learned"],
        default="learned",
        help="Type of positional encoding (default: sinusoidal).",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        default=False,
        help="Disable training, only testing will be done.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        default=False,
        help="Disable testing, only training will be done.",
    )
    parser.add_argument(
        "--no-dropout",
        action="store_true",
        default=False,
        help="Disable dropout in the model.",
    )
    parser.add_argument(
        "--no-batchnorm",
        action="store_true",
        default=True,
        help="Disable batch normalization in the model.",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.2, help="Dropout rate (default: 0.2)."
    )
    parser.add_argument(
        "--train-weight-decay",
        type=float,
        default=0,
        help="Weight decay for SSL (default: 0).",
    )
    parser.add_argument(
        "--test-weight-decay",
        type=float,
        default=0,
        help="Weight decay for SSL (default: 0).",
    )
    parser.add_argument(
        "--activation-function",
        default="sigmoid",
        choices=["sigmoid", "hard_sigmoid", "gumbel_sigmoid"],
        help="Activation function (default: sigmoid).",
    )
    parser.add_argument(
        "--gumbel-temperature",
        type=float,
        default=3.0,
        help="Temperature for gumbel sigmoid (default: 1.0).",
    )
    parser.add_argument(
        "--gumbel-min-temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel sigmoid (default: 0.5).",
    )
    parser.add_argument(
        "--gumbel-annealing-rate",
        type=float,
        default=0.99,
        help="Annealing rate for gumbel sigmoid (default: 0.99).",
    )
    parser.add_argument(
        "--hidden-activation-function",
        default="relu",
        choices=[
            "relu",
            "leaky_relu",
        ],
        help="Activation function (default: relu).",
    )

    # Batch and Epoch Settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Input batch size for training (default: 512).",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        help="Input batch size for testing (default: 2048).",
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train (default: 1)."
    )
    # Learning Rate and Scheduler
    parser.add_argument(
        "--train-lr", type=float, default=1e-3, help="Learning rate (default: 0.001)."
    )
    parser.add_argument(
        "--test-lr", type=float, default=1e-3, help="Learning rate (default: 0.001)."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=False,
        help="Replace the model and outputs. If False then we skip the experiment if the run exists.",
    )
    parser.add_argument(
        "--add-gradient-clipping",
        action="store_true",
        default=False,
        help="Add gradient clipping.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=5.0,
        help="Gradient clipping norm (default: 0.25).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90,
        help="Learning rate step gamma (default: 0.90).",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="ReduceLROnPlateau",
        choices=[
            "StepLR",
            "ReduceLROnPlateau",
            "OneCycleLR",
            "MultiStepLR",
            "ExponentialLR",
            "CosineAnnealingLR",
            "CyclicLR",
            "CosineAnnealingWarmRestarts",
            "None",
        ],
        help="LR Scheduler (default: step).",
    )
    parser.add_argument(
        "--train-optimizer", default="adam", help="Choose the optimizer"
    )
    parser.add_argument("--test-optimizer", default="adam", help="Choose the optimizer")
    # Dataset and Paths
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
    )
    parser.add_argument(
        "--not-use-pgm-with-fewer-cliques",
        dest="use_pgm_with_fewer_cliques",
        action="store_false",
        default=True,
        help="Use PGM with fewer cliques.",
    )
    parser.add_argument(
        "--pgm-model-directory",
        type=str,
        help="Location of the PGM model. Set automatically given PGM.",
    )
    parser.add_argument(
        "--dataset-directory",
        type=str,
        help="Location of the datasets.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="./experiments",
        help="Location of the saved models and outputs.",
    )
    parser.add_argument(
        "--nn-model-path", type=str, help="Location of the trained NN model."
    )
    parser.add_argument(
        "--use-saved-buckets",
        action="store_true",
        default=False,
        help="Use saved buckets.",
    )
    parser.add_argument(
        "--saved-buckets-directory",
        type=str,
        default="",
        help="Directory for saved buckets.",
    )
    parser.add_argument(
        "--same-bucket-iter",
        action="store_true",
        default=False,
        help="Use the same bucket for an epoch.",
    )
    # Miscellaneous
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        help="Batches to wait before logging training status (default: 2).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Threshold for binary output conversion (default: 0.50).",
    )
    parser.add_argument(
        "--threshold_type",
        type=str,
        default="basic",
        choices=[
            "basic",
            "high_uncertainty",
            "knearest_binary_vectors",
            "branch_and_bound",
            "knearest_binary_vectors,branch_and_bound",
        ],
        help="Threshold type (default: basic).",
    )
    parser.add_argument(
        "--uncertainity_max_vars",
        type=int,
        default=10,
        help="Number of max variables for high uncertainty thresholding (default: 10).",
    )
    parser.add_argument(
        "--k_nearest_k",
        type=int,
        default=500,
        help="Number of nearest neighbors (beam width) for finding nearest binary vectors (default: 1000).",
    )
    parser.add_argument(
        "--uncertainity_branch_bound_max_vars",
        type=int,
        default=100,
        help="Max number of variables for branch and bound (default: 100).",
    )
    parser.add_argument(
        "--branch_bound_program_path",
        type=str,
        default="",
        help="Path to the branch and bound program (default: None).",
    )
    parser.add_argument(
        "--use_max_half_vars_branch_bound",
        action="store_true",
        default=False,
        help="Use max half vars branch bound.",
    )
    parser.add_argument(
        "--load_saved_model_for_binarization",
        action="store_true",
        default=False,
        help="Load saved model for binarization. Please provide the path to the saved model (.pt file path).",
    )
    parser.add_argument(
        "--load_saved_outputs_for_binarization",
        action="store_true",
        default=False,
        help="Load saved outputs for binarization.",
    )
    parser.add_argument(
        "--infer_cfg_from_saved_outputs",
        action="store_true",
        default=False,
        help="Infer the cfg from the saved outputs.",
    )
    parser.add_argument(
        "--binarization_saved_output_path",
        type=str,
        default="",
        help="Path to the saved model for binarization. Please provide the path to the saved outputs directory (npz file path) (default: None).",
    )
    parser.add_argument(
        "--binarization_saved_model_path",
        type=str,
        default="",
        help="Path to the saved model for binarization (default: None).",
    )
    parser.add_argument(
        "--query-prob",
        type=float,
        default=0.70,
        help="Probability of query variables (default: 0.70).",
    )
    parser.add_argument(
        "--no-log-loss",
        action="store_true",
        default=False,
        help="Disable log loss usage.",
    )
    parser.add_argument(
        "--add-distance-loss-evid-ll",
        action="store_true",
        default=False,
        help="Use distance loss for evidence LL scores.",
    )
    parser.add_argument(
        "--add-evid-loss",
        action="store_true",
        default=False,
        help="Use distance loss on evidence variables.",
    )
    parser.add_argument(
        "--evid-lambda",
        type=float,
        default=0.1,
        help="Multiplier for distance loss (default: 0.1).",
    )
    parser.add_argument(
        "--add-entropy-loss",
        action="store_true",
        default=False,
        help="Add entropy loss to the model.",
    )
    parser.add_argument(
        "--entropy-lambda",
        type=float,
        default=0.01,
        help="Multiplier of the entropy loss (default: 0.01).",
    )

    parser.add_argument(
        "--num-train-examples",
        type=int,
        default=0,
        help="Number of training examples to use (default: 10000).",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 5).",
    )

    parser.add_argument(
        "--train-on-test-set",
        action="store_true",
        default=False,
        help="Train on test set for self-supervised methods.",
    )
    parser.add_argument(
        "--only-test-train-on-test-set",
        action="store_true",
        default=False,
        help="Only test on train on test set.",
    )
    parser.add_argument(
        "--train-on-test-set-scheduler",
        choices=["StepLR", "None"],
        default="None",
        help="LR Scheduler (default: step).",
    )

    parser.add_argument(
        "--duplicate-example-train-on-test",
        action="store_true",
        default=False,
        help="Duplicate examples for training on test set - create a batch with single example.",
    )
    parser.add_argument(
        "--perturb-model-train-on-test",
        action="store_true",
        default=False,
        help="Perturb the model.",
    )
    parser.add_argument(
        "--num-init-train-on-test",
        type=int,
        default=1,
        help="Number of times to initialize the model (default: 1).",
    )
    parser.add_argument(
        "--num-iter-train-on-test",
        type=int,
        default=500,
        help="Number of epochs to train on the test set (default: 5).",
    )
    parser.add_argument(
        "--use-batch-train-on-test",
        action="store_true",
        default=False,
        help="Use batch training on test set.",
    )

    parser.add_argument(
        "--prev-threshold-nn-ll",
        type=float,
        default=0.0,
        help="Previous threshold NN LL score (default: 0.0).",
    )
    parser.add_argument(
        "--num-test-examples",
        type=int,
        default=2000,
        help="Number of test examples to use (default: 0).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["mpe"],
        default="mpe",
        help="Task type (default: mpe).",
    )
    parser.add_argument(
        "--partition_type",
        type=str,
        choices=["anyPartitionMPE", "fixedPartitionMPE"],
        default="anyPartitionMPE",
        help="Task type (default: fixedPartitionMPE).",
    )
    parser.add_argument(
        "--not-save-model",
        action="store_true",
        default=False,
        help="Disable saving the current model.",
    )
    parser.add_argument(
        "--no-extra-data",
        action="store_true",
        default=True,
        help="Do not use extra data.",
    )
    parser.add_argument(
        "--debug-tuning",
        action="store_true",
        default=False,
        help="Find the best hyperparameters and model architecture.",
    )

    cfg = parser.parse_args()
    cfg.no_extra_data = True
    return process_arguments(cfg)


def process_arguments(cfg):
    get_dataset_type(cfg)
    if cfg.no_train:
        ensure_training_setup(cfg)
        update_model_path(cfg)
        validate_model_and_buckets(cfg)

    if cfg.use_saved_buckets:
        use_saved_buckets(cfg)

    if should_update_model_path(cfg):
        update_nn_model_path(cfg)

    if cfg.add_entropy_loss:
        log_entropy_loss_setup(cfg)

    log_query_probability_setup(cfg)
    calculate_probabilities(cfg)

    validate_probabilities(cfg)
    add_time_to_cfg(cfg)

    project_name = generate_project_name(cfg)
    return cfg, project_name


def get_dataset_type(cfg):
    # get dataset type if present in pairwise_datasets or complex_datasets
    # else keep the value as is
    if cfg.dataset in pairwise_datasets:
        cfg.pgm = "mn_pairwise"
    elif cfg.dataset in complex_datasets:
        cfg.pgm = "mn_higher_order"


def add_time_to_cfg(cfg):
    cfg.time = time.strftime("%Y%m%d-%H%M%S")


def ensure_training_setup(cfg):
    assert (
        (cfg.no_train == cfg.use_saved_buckets)
        or (cfg.no_train == cfg.only_test_train_on_test_set)
        or (cfg.no_train == cfg.load_saved_outputs_for_binarization)
        or (cfg.no_train == cfg.load_saved_model_for_binarization)
    ), "We need buckets to test the model"
    logger.info("Not training the model")


def update_model_path(cfg):
    cfg.nn_model_path = os.path.join(
        cfg.saved_buckets_directory.replace("model_outputs", "models"), "model.pt"
    )
    if "nn_merge" in cfg.nn_model_path:
        cfg.nn_model_path = cfg.nn_model_path.replace("nn_merge/", "")
    logger.info(f"Using trained model from the directory: {cfg.nn_model_path}")


def validate_model_and_buckets(cfg):
    assert (
        cfg.nn_model_path is not None
    ), "Please provide the model directory for a trained NN model"
    assert (
        cfg.use_saved_buckets
        or cfg.only_test_train_on_test_set
        or cfg.load_saved_outputs_for_binarization
        or cfg.load_saved_model_for_binarization
    ), "Please provide previously saved buckets path - we use the same buckets for testing"


def use_saved_buckets(cfg):
    logger.info("Using saved buckets")
    logger.info(f"Saved buckets directory: {cfg.saved_buckets_directory}")
    logger.info("Please note using this might delete old models and outputs")
    update_task_and_dataset(cfg)


def should_update_model_path(cfg):
    return (
        "mnist" in cfg.saved_buckets_directory or "cifar" in cfg.saved_buckets_directory
    ) and cfg.no_train


def update_nn_model_path(cfg):
    pattern = r"/models_mpe_\w+/"
    replacement = "/models_mpe/"
    cfg.nn_model_path = re.sub(pattern, replacement, cfg.nn_model_path)


def log_entropy_loss_setup(cfg):
    logger.info("Using entropy loss")
    logger.info(f"Entropy lambda: {cfg.entropy_lambda}")


def log_query_probability_setup(cfg):
    logger.info(
        "Please set the value of evidence and we can calculate the query probability"
    )


def calculate_probabilities(cfg):
    if cfg.task == "mpe":
        cfg.evidence_prob = 1 - cfg.query_prob
        cfg.others_prob = 0
    elif cfg.task == "mmap":
        cfg.evidence_prob = 1 - cfg.query_prob
        cfg.others_prob, cfg.evidence_prob = (
            cfg.evidence_prob / 2,
            cfg.evidence_prob / 2,
        )
    else:
        raise ValueError("Please select a task")


def validate_probabilities(cfg):
    tolerance = 0.0001
    assert (
        abs(cfg.query_prob + cfg.evidence_prob + cfg.others_prob - 1) < tolerance
    ), f"{cfg.query_prob} + {cfg.evidence_prob} + {cfg.others_prob} != 1"


def generate_project_name(cfg):
    project_name = f"Adv_Model_T-{cfg.task}_PT-{cfg.partition_type}_PGM-{cfg.pgm}_D-{cfg.dataset}_M-{cfg.model}_MT-{cfg.model_type}_QP-{round(cfg.query_prob, 2)}_EP-{round(cfg.evidence_prob, 2)}"
    if cfg.no_train:
        project_name = "only_test_" + project_name
    return project_name


def update_task_and_dataset(cfg):
    # Split the directory string into parts based on underscores
    parts = cfg.saved_buckets_directory.split("_")

    # Initialize variables to hold the probabilities
    evidence_prob_str = ""
    query_prob_str = ""

    # Iterate over the parts to find and extract the relevant values
    for part in parts:
        if part.startswith("EvidProb-"):
            evidence_prob_str = part.split("-")[-1]
        elif part.startswith("QueryProb-"):
            query_prob_str = part.split("-")[-1]

    # Convert the extracted string values to float
    cfg.evidence_prob = float(evidence_prob_str) if evidence_prob_str else 0.0
    cfg.query_prob = float(query_prob_str) if query_prob_str else 0.0
    update_task_from_directory(cfg)


def update_task_from_directory(cfg):
    if "mmap" in cfg.saved_buckets_directory.lower():
        cfg.task = "mmap"
    elif "mpe" in cfg.saved_buckets_directory.lower():
        cfg.task = "mpe"
