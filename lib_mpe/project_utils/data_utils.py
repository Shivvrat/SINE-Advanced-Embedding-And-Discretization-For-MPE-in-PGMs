import os
import pickle
import random
import re

import numpy as np
import torch
from loguru import logger
from mpe_advanced_models.model_1.src.data.collate_funcs import (
    collate_hypergraph_graph_data,
)
from torch.distributions import Categorical


def init_train_test_data(
    cfg, test_data, test_buckets, val_data, val_buckets, train_data
):
    if cfg.num_test_examples:
        # We are only using a subset of the test data
        logger.info(f"Using only {cfg.num_test_examples} test examples")
        test_data = test_data[: cfg.num_test_examples]
        test_buckets = {
            key: test_buckets[key][: cfg.num_test_examples] for key in test_buckets
        }
    if cfg.num_train_examples:
        # We are only using a subset of the train data
        logger.info(f"Using only {cfg.num_train_examples} train examples")
        train_data = train_data[: cfg.num_train_examples]
    num_val_examples = min(val_data.shape[0], val_buckets["evid"].shape[0])
    val_data = val_data[:num_val_examples]
    val_buckets = {key: val_buckets[key][:num_val_examples] for key in val_buckets}
    return test_data, test_buckets, train_data, val_data, val_buckets


def init_dataloader_args(cfg, use_cuda):
    train_kwargs = {"batch_size": cfg.batch_size}
    test_kwargs = {"batch_size": cfg.test_batch_size}
    if use_cuda:
        if cfg.data_device == "cuda":
            cuda_kwargs = {
                "num_workers": 0,
                "pin_memory": False,
            }
        else:
            cuda_kwargs = {
                "num_workers": 4,
                "pin_memory": True,
            }

        train_kwargs = {**train_kwargs, **cuda_kwargs, "shuffle": True}
        test_kwargs = {**test_kwargs, **cuda_kwargs}
        # test_kwargs |= cuda_kwargs
        test_kwargs["shuffle"] = False
    return train_kwargs, test_kwargs


def ensure_all_buckets(buckets, sample_size):
    """
    Ensures all required buckets are present, initializing them if necessary.
    """
    for bucket_name in ["evid", "query", "unobs"]:
        buckets.setdefault(bucket_name, torch.zeros(sample_size, dtype=torch.bool))


def create_buckets(n, probabilities):
    """
    Distributes numbers from 0 to n-1 into buckets based on given probabilities.

    cfg:
        n (int): The range of numbers (0 to n-1) to be divided.
        probabilities (list): List of probabilities for each bucket, summing up to 1.0.

    Returns:
        dict: A dictionary where each key corresponds to a bucket ('evid', 'query', 'unobs')
              and the value is a boolean tensor indicating the indices belonging to that bucket.

    Example:
        >>> n = 10
        >>> probabilities = [0.4, 0.3, 0.3]
        >>> buckets = create_buckets(n, probabilities)
        >>> print(buckets)
        {'evid': tensor([True, False, False, True, False, True, False, True, False, True]),
         'query': tensor([False, True, True, False, True, False, False, False, False, False]),
         'unobs': tensor([False, False, False, False, False, False, True, False, True, False])}
    """
    # Sample from the distribution to assign each number to a bucket
    distribution = Categorical(torch.tensor(probabilities))
    samples = distribution.sample(torch.Size([n]))

    # Initialize a dictionary to hold the buckets
    buckets = {
        "evid": torch.zeros(n, dtype=torch.bool),
        "query": torch.zeros(n, dtype=torch.bool),
        "unobs": torch.zeros(n, dtype=torch.bool),
    }

    # Populate each bucket based on the sampled indices
    for i, bucket_name in enumerate(buckets.keys()):
        buckets[bucket_name] = samples == i

    return buckets


def count_flipped_bits(array):
    """
    The function `count_flipped_bits` takes an array of binary values, computes the XOR of each row with
    its subsequent row, and counts the number of flipped bits in each XOR result.

    :param array: The input parameter "array" is expected to be a numpy array containing binary values.
    Each row of the array represents a binary number
    :return: The function `count_flipped_bits` returns an array that contains the number of flipped bits
    between each row and its subsequent row in the input array.
    """
    array = array.astype(np.uint8)
    # Compute the XOR of each row with its subsequent row
    xor_result = np.bitwise_xor(array[:-1], array[1:])

    # Count the number of set bits in each XOR result
    flipped_bits = np.unpackbits(xor_result, axis=1).sum(axis=1)

    return flipped_bits


def get_dataloaders(
    cfg,
    train_data,
    test_data,
    val_data,
    train_buckets,
    val_buckets,
    test_buckets,
    use_cuda,
    unique_buckets,
    torch_pgm,
    num_variables_in_buckets,
    pretrained_features=None,
):
    from mpe_advanced_models.model_1.src.data.dataloaders import (
        MMAPTestDataset,
        MMAPTrainDataset,
    )

    train_dataset = MMAPTrainDataset(
        data=train_data,
        buckets=train_buckets,
        num_var_in_buckets=num_variables_in_buckets,
        pgm=torch_pgm,
        model_type=cfg.model,
        data_device=cfg.data_device,
        input_type=cfg.input_type,
        embedding_type=cfg.embedding_type,
        same_bucket_for_iter=cfg.same_bucket_iter,
        use_single_model=cfg.use_single_model,
        unique_buckets=unique_buckets,
        task=cfg.task,
        partition_type=cfg.partition_type,
        use_static_node_features=cfg.use_static_node_features,
    )
    test_dataset = MMAPTestDataset(
        data=test_data,
        buckets=test_buckets,
        num_var_in_buckets=num_variables_in_buckets,
        pgm=torch_pgm,
        model_type=cfg.model,
        data_device=cfg.data_device,
        input_type=cfg.input_type,
        embedding_type=cfg.embedding_type,
        partition_type=cfg.partition_type,
        use_static_node_features=cfg.use_static_node_features,
    )
    val_dataset = MMAPTestDataset(
        data=val_data,
        buckets=val_buckets,
        num_var_in_buckets=num_variables_in_buckets,
        pgm=torch_pgm,
        model_type=cfg.model,
        data_device=cfg.data_device,
        input_type=cfg.input_type,
        embedding_type=cfg.embedding_type,
        partition_type=cfg.partition_type,
        use_static_node_features=cfg.use_static_node_features,
    )
    logger.info(f"Test data shape: {test_data.shape}")
    logger.info(f"Test bucket shape: {test_buckets['evid'].shape}")
    logger.info(f"Val data shape: {val_data.shape}")
    logger.info(f"Val bucket shape: {val_buckets['evid'].shape}")
    train_kwargs, test_kwargs = init_dataloader_args(cfg, use_cuda)
    logger.info(f"You have selected {cfg.dataset}")
    # Use train kwargs for train and test_kwcfg for val, test
    if cfg.embedding_type == "hgnn":
        train_kwargs["collate_fn"] = collate_hypergraph_graph_data
        test_kwargs["collate_fn"] = collate_hypergraph_graph_data
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    return train_loader, test_loader, val_loader


def get_num_var_in_buckets(cfg, num_outputs):
    if cfg.use_single_model:
        num_variables_in_buckets = [None, None, None]
    else:
        probabilities = [cfg.evidence_prob, cfg.query_prob, cfg.others_prob]
        num_variables_in_buckets = [
            int(num_outputs * probabilities[0]),
            int(num_outputs * probabilities[1]),
            int(num_outputs * probabilities[2]),
        ]
        if sum(num_variables_in_buckets) != num_outputs:
            # make sure the sum of the buckets is equal to the number of features
            num_variables_in_buckets[0] += num_outputs - sum(num_variables_in_buckets)
    return num_variables_in_buckets


def load_data(cfg, dataset_name):
    (
        data,
        extra_data,
        buckets,
    ) = load_dataset_bn_mn(
        dataset_name,
        cfg,
    )

    return data, extra_data, buckets


def load_dataset_bn_mn(
    dataset_name,
    cfg,
):
    """
    Load training, testing, and validation data from a specified dataset,
    and optionally convert to torch tensors and load additional data and test buckets.

    :param dataset_name: Name of the dataset.
    :param cfg: Configuration settings for data loading.
    :param is_torch: Convert data to torch tensors if True.
    :param load_test_buckets: Load test buckets if True.
    :return: Tuple containing loaded data.
    """
    path_to_datasets = cfg.dataset_directory
    data_types = ["train", "test", "valid", "val"]

    # Function to load data from a file
    def load_data(file_path):
        if os.path.exists(file_path):
            try:
                return np.loadtxt(
                    file_path,
                )
            except:
                return np.genfromtxt(
                    file_path, delimiter=" ", autostrip=True, dtype=np.int32
                )
        return None

    # Loading data
    data = {}
    for dtype in data_types:
        file_path = os.path.join(path_to_datasets, dataset_name, f"{dtype}.txt")
        data[dtype] = load_data(file_path)

    # Handling missing validation data
    if data["valid"] is None and data["val"] is not None:
        data["valid"] = data["val"]

    # Loading extra data if available and not disabled in cfg
    extra_data = np.array([])

    # Converting to torch tensors if required
    for dtype in data:
        if data[dtype] is not None:
            data[dtype] = torch.from_numpy(data[dtype]).double()
    extra_data = torch.from_numpy(extra_data).float()

    # Load test buckets if required
    logger.info("Loading buckets")
    buckets_path = path_to_datasets.replace(
        "sampled_data", f"sampled_data_buckets_{cfg.partition_type}"
    )
    buckets_path = os.path.join(
        buckets_path,
        "mpe",
        dataset_name,
        f"evid-{round(cfg.evidence_prob, 2)}_query-{round(cfg.query_prob, 2)}",
    )
    if cfg.partition_type == "anyPartitionMPE":
        bucket_files = [
            f"{dataset_name}.{dtype}.buckets.npz" for dtype in ["test", "valid", "val"]
        ]
        buckets = {}
        for bfile in bucket_files:
            try:
                buckets[bfile.split(".")[-3]] = np.load(
                    os.path.join(buckets_path, bfile)
                )
            except FileNotFoundError:
                continue

        # Handling missing validation buckets
        if "valid" not in buckets and "val" in buckets:
            buckets["valid"] = buckets["val"]
        buckets["train"] = None
    elif cfg.partition_type == "fixedPartitionMPE":
        bfile = "buckets.npz"
        buckets = np.load(os.path.join(buckets_path, bfile))
        final_buckets = {"test": {}, "valid": {}, "train": {}}

        for partition_key in buckets:
            for dataset_type in final_buckets:
                # repeat the single row in bucket number of examples in corresponding data
                final_buckets[dataset_type][partition_key] = np.tile(
                    buckets[partition_key],
                    (data[dataset_type].shape[0], 1),
                )
        buckets = final_buckets
    return (data, extra_data, buckets)


def get_mpe_solutions(
    cfg,
):
    if cfg.pgm in ["bn", "mn_pairwise", "mn_higher_order"]:
        mpe_solutions = load_mpe_solutions_mn_bn(cfg.dataset, cfg)
    return mpe_solutions


def load_mpe_solutions_mn_bn(
    dataset_name,
    cfg,
):
    path_to_datasets = os.path.dirname(cfg.dataset_directory)
    "mn/data/pairwise/"
    head, sep, tail = path_to_datasets.rpartition("/data/")
    new_path = head + f"/baseline_{cfg.baseline}_solutions/" + tail
    mpe_solutions_path = os.path.join(
        new_path,
        cfg.partition_type,
        dataset_name,
        f"evid-{round(cfg.evidence_prob, 2)}_query-{round(cfg.query_prob, 2)}",
    )
    mpe_solutions_path = os.path.join(
        mpe_solutions_path,
        f"{dataset_name}.test.solutions.npz",
    )
    if os.path.exists(mpe_solutions_path):
        test_data = np.load(
            mpe_solutions_path,
        )
        test_mpe_output = test_data["mpe_outputs"]
        try:
            test_root_ll_pgm = test_data["ll_scores"]
        except ValueError:
            test_data = np.load(mpe_solutions_path, allow_pickle=True)
            test_root_ll_pgm = test_data["ll_scores"]
            test_mpe_output = test_data["mpe_outputs"]
    else:
        test_mpe_output = None
        test_root_ll_pgm = -np.inf

    outputs = {
        "test_mpe_output": test_mpe_output,
        "test_root_ll_pgm": test_root_ll_pgm,
    }
    return outputs


def divide_randomly(value):
    """
    The function `divide_randomly` takes a value between 0 and 1, and divides it randomly into two parts
    with two decimal points.

    :param value: The value parameter represents a number between 0 and 1
    :return: The function `divide_randomly` returns a tuple containing two values: `first_part` and
    `second_part`.
    """
    # Ensure the value is between 0 and 1
    if value < 0 or value > 1:
        raise ValueError("Value should be between 0 and 1.")

    # Generate a random number with two decimal points between 0 and 1/4 of the value
    first_part = random.uniform(value / 4, value / 2)

    # Calculate the second part with two decimal points by rounding it to avoid precision issues
    second_part = value - first_part

    return first_part, second_part
