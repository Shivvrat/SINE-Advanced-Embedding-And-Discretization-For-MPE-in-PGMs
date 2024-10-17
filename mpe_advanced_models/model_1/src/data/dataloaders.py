import numpy as np
import torch
from loguru import logger
from mpe_advanced_models.model_1.src.data.create_buckets import (
    create_buckets_all_queries_per,
    create_buckets_one_query_per,
)
from mpe_advanced_models.model_1.src.data.pyg_data import HyperGraphData
from torch.utils.data import Dataset


def process_bucket_for_nn_discrete(sample, buckets, pgm=None, node_features=None):
    """
    Process bucket for neural network with discrete samples.
    Handles both single example (1D tensor) and batch of examples (2D tensor).
    """

    # Check if the sample is a batch (2D) or a single example (1D)
    if sample.dim() == 1:
        n_vars = sample.size(0)
        final_sample = torch.zeros(n_vars * 2, dtype=sample.dtype, device=sample.device)

        final_sample[0::2] = sample.double()  # Assign even indices
        final_sample[1::2] = 1 - sample.double()  # Assign odd indices

        for key, value in (("query", 0), ("unobs", 1)):
            mask = buckets[key]
            final_sample[::2][mask] = final_sample[1::2][mask] = value

    elif sample.dim() == 2:
        num_samples, n_vars = sample.size()
        final_sample = torch.zeros(
            num_samples, n_vars * 2, dtype=sample.dtype, device=sample.device
        )

        final_sample[:, 0::2] = sample.double()  # Assign even indices to all samples
        final_sample[:, 1::2] = 1 - sample.double()  # Assign odd indices to all samples

        # Apply masks for 'query' and 'unobserved' buckets
        for key, value in (("query", 0), ("unobs", 1)):
            mask = buckets[key]
            final_sample[:, ::2][mask] = final_sample[:, 1::2][mask] = value

    else:
        raise ValueError(f"Invalid sample dimension: {sample.dim()}")
    return {"embedding": final_sample}


def process_bucket_for_hypergraph(sample, buckets, pgm, node_features):
    """
    Process bucket for neural network with discrete samples.
    Handles both single example (1D tensor) and batch of examples (2D tensor).
    """

    # Check if the sample is a batch (2D) or a single example (1D)
    if sample.dim() == 1:
        (
            instantiated_factors,
            instantiated_hyperedge_index,
            instantiated_vars,
            instantiated_domains,
        ) = pgm.initialize_evidence_cython(
            sample.unsqueeze(0).numpy(),
            buckets["evid"].unsqueeze(0).numpy(),
        )
        # makes values -1 if they are in buckets["query"]
        sample[buckets["query"]] = -1
        sample = torch.cat((sample.unsqueeze(1), buckets["query"].unsqueeze(1)), dim=1)
        if node_features is not None:
            # node_features is a tensor of shape (num_nodes, num_features)
            # with the features for each node: first degree, then clustering, then betweenness, then closeness, then eigenvector
            sample = torch.cat((sample, node_features), dim=1)
    else:
        raise ValueError(f"Invalid sample dimension: {sample.dim()}")
    data = HyperGraphData(
        x=sample,
        edge_index=torch.from_numpy(instantiated_hyperedge_index),
        edge_attr=torch.from_numpy(
            instantiated_factors.squeeze(0),
        ).to(sample.dtype),
    )
    return {
        "graph_data": data,
    }


def create_attention_mask_for_transformer(buckets):
    attention_mask = torch.zeros_like(buckets["evid"], dtype=torch.float)
    # Set the positions of unobs to -inf
    # Since we don't know the value of unobs, we don't want to attend to it
    attention_mask[buckets["unobs"]] = float("-inf")

    return attention_mask


class MMAPBaseDataset(Dataset):
    """
    Base dataset class for MMAP datasets, providing common functionalities.
    """

    def __init__(
        self,
        data,
        pgm=None,
        model_type="nn",
        data_device="cuda",
        input_type="data",
        embedding_type="hgnn",
        partition_type="fixedPartitionMPE",
        use_static_node_features=False,
    ):
        self.data = data.double()  # .to(data_device) -> won't work for hgnn
        self.pgm = pgm
        self.model_type = model_type
        self.input_type = input_type
        self.partition_type = partition_type
        self.embedding_type = embedding_type
        self.process_bucket_function = self.determine_process_function(model_type)
        if use_static_node_features:
            self.node_features = self.get_node_features()
        else:
            self.node_features = None

    def get_node_features(self):
        """
        Get the node features for the given PGM.
        """
        import networkx as nx
        from lib_mpe.models.encoder.graph_encodings import NodeFeatureExtractor

        cliques_list = self.pgm.pgm.cliques
        num_nodes = self.pgm.pgm.num_vars
        # convert cliques_list to a adjacency matrix
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for clique in cliques_list:
            for node in clique:
                for other_node in clique:
                    if node != other_node:
                        adjacency_matrix[node, other_node] = 1

        # initialize networkx graph
        networkx_pgm = nx.from_numpy_array(adjacency_matrix)
        node_feature_extractor = NodeFeatureExtractor(networkx_pgm)
        # get the node features
        node_features = node_feature_extractor.get_features()
        return torch.from_numpy(node_features).double()

    def __len__(self):
        return len(self.data)

    def determine_process_function(self, model_type):
        """
        Determines the appropriate processing function based on the model type.
        """
        if model_type in ["nn", "transformer"]:
            if self.embedding_type in ["discrete"]:
                return process_bucket_for_nn_discrete
            elif self.embedding_type in ["hgnn"]:
                assert self.partition_type in ["fixedPartitionMPE", "anyPartitionMPE"]
                return process_bucket_for_hypergraph
            elif self.embedding_type in ["gnn"]:
                raise NotImplementedError("GNN is not supported yet")
            else:
                raise ValueError(f"Invalid embedding type: {self.embedding_type}")
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def create_buckets(self, sample, index):
        """
        Creates buckets for the given sample. To be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def __getitem__(self, index):
        sample = self.data[index]
        buckets = self.create_buckets(sample, index)
        processed_sample = self.process_bucket_function(
            sample,
            buckets,
            self.pgm,
            self.node_features,
        )
        attention_mask = self.create_attention_mask(buckets, sample)
        return_dict = {
            "index": index,
            "initial_data": sample,
            "attention_mask": attention_mask,
            **buckets,
            **processed_sample,
        }
        return return_dict

    def create_attention_mask(self, buckets, sample):
        """
        Creates an attention mask based on the model type.
        """
        return torch.zeros_like(sample, dtype=torch.float)


class MMAPTrainDataset(MMAPBaseDataset):
    """
    Dataset class for MMAP training data.
    """

    def __init__(
        self,
        data,
        buckets,
        num_var_in_buckets: list = None,
        same_bucket_for_iter: bool = False,
        use_single_model: bool = False,
        task: str = "mpe",
        unique_buckets=None,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.num_var_in_buckets = num_var_in_buckets
        self.same_bucket_for_iter = same_bucket_for_iter
        self.use_single_model = use_single_model
        self.unique_buckets = unique_buckets
        self.task = task
        self.partition_type = kwargs["partition_type"]

        if unique_buckets is not None:
            self.num_unique_buckets = len(unique_buckets["query"])
        else:
            self.num_unique_buckets = -1

        if self.partition_type == "fixedPartitionMPE":
            logger.info("Using fixed partition")
            self.setup_buckets_fixed_partition(buckets, kwargs["data_device"])
            self.create_buckets = self.create_buckets_fixed_partition
        elif self.partition_type == "anyPartitionMPE":
            logger.info("Using any partition")
            self.create_buckets = self.create_buckets_any_partition
        else:
            raise ValueError(f"Invalid partition type: {self.partition_type}")

    def setup_buckets_fixed_partition(self, buckets, data_device):
        """
        Setup buckets for fixed partition.
        """
        self.buckets = {
            key: torch.from_numpy(value).bool()  # .to(data_device)
            for key, value in buckets.items()
        }
        self.bucket_names = (
            ["evid", "query", "unobs"]
            if "unobs" in buckets and len(buckets["unobs"]) > 0
            else ["evid", "query"]
        )

    def create_buckets_fixed_partition(self, sample, index):
        """
        Creates buckets for fixed partition.
        """
        return {
            bucket_name: self.buckets[bucket_name][index]
            for bucket_name in self.bucket_names
        }

    def create_buckets_any_partition(self, sample, index):
        """
        Creates buckets for training data based on the data distribution.
        """
        if self.same_bucket_for_iter:
            return {
                "evid": torch.zeros(sample.shape[0], dtype=torch.bool),
                "query": torch.zeros(sample.shape[0], dtype=torch.bool),
                "unobs": torch.zeros(sample.shape[0], dtype=torch.bool),
            }
        if self.use_single_model:
            buckets = create_buckets_all_queries_per(
                sample.shape[0], self.task, sample.device
            )
        else:
            buckets = create_buckets_one_query_per(
                sample.shape[0], self.num_var_in_buckets, sample.device
            )
        if not self.same_bucket_for_iter:
            self.ensure_all_buckets(buckets, sample.shape[0])
        return buckets

    @staticmethod
    def ensure_all_buckets(buckets, sample_size):
        """
        Ensures all required buckets are present, initializing them if necessary.
        """
        for bucket_name in ["evid", "query", "unobs"]:
            buckets.setdefault(bucket_name, torch.zeros(sample_size, dtype=torch.bool))


class MMAPTestDataset(MMAPBaseDataset):
    """
    Dataset class for MMAP test data.
    """

    def __init__(
        self,
        data,
        buckets,
        num_var_in_buckets,
        data_device,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.buckets = {
            key: torch.from_numpy(value).bool()  # .to(data_device)
            for key, value in buckets.items()
        }
        self.bucket_names = (
            ["evid", "query", "unobs"]
            if "unobs" in buckets and len(buckets["unobs"]) > 0
            else ["evid", "query"]
        )
        self.num_var_in_buckets = num_var_in_buckets

    def create_buckets(self, sample, index):
        """
        Retrieves the appropriate buckets for the given test sample.
        """
        return {
            bucket_name: self.buckets[bucket_name][index]
            for bucket_name in self.bucket_names
        }
