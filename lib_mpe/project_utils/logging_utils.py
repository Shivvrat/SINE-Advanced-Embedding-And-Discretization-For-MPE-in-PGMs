import os

import torch
import wandb
import yaml
from loguru import logger


def init_logger_and_wandb(project_name, args):
    """
    The function initializes a logger and wandb for a given project name and arguments, and returns the
    device, use_cuda, and use_mps variables.

    :param project_name: The project name is a string that represents the name of the project you are
    working on. It is used to initialize the logging and wandb (Weights & Biases) project
    :param args: The "args" parameter is a dictionary or object that contains various configuration
    options or arguments for the logger and wandb initialization. It is used to update the wandb
    configuration and determine whether to use CUDA or MPS for training
    :return: three values: `device`, `use_cuda`, and `use_mps`.
    """
    import sys

    dataset_name = args.dataset
    task = args.task
    # Initialize wandb with project details
    wandb.init(
        project=f"MMAP-Adv-{dataset_name}-{task}-{args.embedding_type}-{args.model}-{args.query_prob}",
        name=project_name,
        config=args,
    )

    # Configure loguru logger
    config = {
        "handlers": [
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss.SSS}</green> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <lvl>{level}: {message}</lvl>",
                "colorize": True,
            },
            {
                "sink": "logging/{time:YYYY-MM-DD}/"
                + f"{project_name}/"
                + "logger_{time}.log",
                "format": "{time:YYYY-MM-DD at HH:mm:ss.SSS} | {name}:{function}:{line} | {level}: {message}",
                "rotation": "1 week",
                "retention": "1 month",
                "compression": "zip",
            },
        ],
        "extra": {"user": "usr"},
    }

    logger.configure(**config)
    logger.add(
        os.path.join(args.model_outputs_dir, "logs.log"),
        format="{time} {level} {message}",
    )
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    # num_classes = 1
    if use_cuda:
        device = "cuda"
        logger.info("Using GPU for training")

    elif use_mps:
        device = "mps"
        logger.info("Using MPS for training")

    else:
        device = "cpu"
        logger.info("Using CPU for training")
    logger.info(args)
    return device, use_cuda, use_mps


def save_args_as_yaml(args, filename="args.yaml"):
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(filename, "w") as file:
        yaml.dump(args_dict, file, default_flow_style=False)


def load_args_from_yaml(filename="args.yaml"):
    with open(filename, "r") as file:
        args_dict = yaml.load(file, Loader=yaml.FullLoader)
    return args_dict
