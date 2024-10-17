import random

import torch


def create_buckets_one_query_per(n, num_in_buckets, device="cuda"):
    """
    Distributes numbers from 0 to n-1 into buckets with a faster random assignment of indices to True.

    Args:
        n (int): The range of numbers (0 to n-1) to be divided.
        num_in_buckets (list): List of integers representing the number of variables
                               in each bucket.

    Returns:
        dict: A dictionary where each key corresponds to a bucket ('evid', 'query', 'unobs')
              and the value is a boolean tensor indicating the randomly selected indices.

    Example:
        >>> n = 10
        >>> num_in_buckets = [4, 0, 3]
        >>> buckets = create_buckets_random_fast(n, num_in_buckets)
        >>> print(buckets)
        {'evid': tensor([False, False, False, False, True, True, True, True, False, True]),
         'query': tensor([True, False, False, False, False, False, False, False, False, False]),
         'unobs': tensor([False, True, True, True, False, False, False, False, True, False])}
    """
    # Generate a random permutation of indices from 0 to n-1
    random_indices = torch.randperm(n)

    # Initialize a dictionary to hold the buckets
    buckets = {
        "evid": torch.zeros(n, dtype=torch.bool, device=device),
        "query": torch.zeros(n, dtype=torch.bool, device=device),
        "unobs": torch.zeros(n, dtype=torch.bool, device=device),
    }

    # Iterate through each bucket and assign random indices to True
    start_idx = 0
    for bucket_name, num_vars in zip(buckets.keys(), num_in_buckets):
        # Check if the bucket has zero variables
        if num_vars == 0:
            continue

        # Assign the first num_vars indices to True in the bucket
        end_idx = start_idx + num_vars
        selected_indices = random_indices[start_idx:end_idx]
        buckets[bucket_name][selected_indices] = True
        start_idx = end_idx

    return buckets


def get_bucket_from_unique_set(unique_buckets, num_unique_buckets, task, device="cuda"):
    random_idx = random.randint(0, num_unique_buckets - 1)
    buckets = {
        "evid": torch.from_numpy(unique_buckets["evid"][random_idx]).bool().to(device),
        "query": torch.from_numpy(unique_buckets["query"][random_idx])
        .bool()
        .to(device),
    }
    if task == "mpe":
        buckets["unobs"] = (
            torch.from_numpy(unique_buckets["unobs"][0]).bool().to(device)
        )
    elif task == "mmap":
        buckets["unobs"] = (
            torch.from_numpy(unique_buckets["unobs"][random_idx]).bool().to(device)
        )
    return buckets


def create_buckets_all_queries_per(n, task="mpe", device="cuda"):
    # Calculate the minimum amount for each part based on 10 percent
    min_amount = max(1, n // 10)

    # Randomly distribute the remaining amount
    remaining = n - 3 * min_amount
    random_addition = random.randint(0, remaining)

    # Divide n into two parts with randomness
    num_var_in_query = 2 * min_amount + random_addition
    num_var_in_evid = n - num_var_in_query

    # Adjust the division for tasks other than "mpe"
    if task != "mpe":
        half_evid = num_var_in_evid // 2
        num_var_in_evid, num_var_in_unobs = half_evid, half_evid
    else:
        num_var_in_unobs = 0

    # Ensure that the total number of variables is n
    num_var_in_evid += n - (num_var_in_query + num_var_in_evid + num_var_in_unobs)

    # Prepare bucket values
    num_in_buckets = [num_var_in_evid, num_var_in_query, num_var_in_unobs]
    return create_buckets_one_query_per(n, num_in_buckets, device=device)
