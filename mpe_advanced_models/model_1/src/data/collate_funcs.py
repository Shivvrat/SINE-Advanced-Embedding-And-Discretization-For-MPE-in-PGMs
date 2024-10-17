import torch
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Batch


def collate_sampled_buckets(batch):
    """
    Custom collate function to process batches of data where each item is a dictionary.
    Ensures the output for each key is a 2D tensor.
    Concatenates 1D tensors along a new dimension and stacks 2D tensors along the first dimension.

    Args:
        batch (list of dicts): List of dictionaries with tensors.

    Returns:
        dict: A dictionary with 2D tensors.
    """
    # Initialize a dictionary to hold the collated data
    collated_batch = {}

    # Iterate over keys in the dictionary
    for key in batch[0].keys():
        # Check the dimension of the first item for this key to determine processing
        if batch[0][key].dim() == 1:
            # If the tensors are 1D, stack them along a new dimension to make them 2D
            collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
        elif batch[0][key].dim() == 2:
            # If the tensors are 2D, concatenate them along the first dimension
            collated_batch[key] = torch.cat([item[key] for item in batch], dim=0)
        else:
            raise ValueError("Tensors must be either 1D or 2D.")

    return collated_batch


def collate_hypergraph_graph_data(batch):
    """
    Custom collate function to process batches of data where each item is a dictionary.
    Ensures the output for each key is a 2D tensor.
    Concatenates 1D tensors along a new dimension and stacks 2D tensors along the first dimension.

    Args:
        batch (list of dicts): List of dictionaries with tensors.
    """
    collated_batch = {}
    for key in batch[0].keys():
        if key == "graph_data":
            # Use Batch.from_data_list for graph data
            collated_batch[key] = Batch.from_data_list([item[key] for item in batch])
        else:
            # Use default_collate for other data
            collated_batch[key] = default_collate([item[key] for item in batch])

    return collated_batch
