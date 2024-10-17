import torch
from dataloaders import create_buckets_one_query_per


# Check if the values in the tensors of both dictionaries are equal
def compare_buckets(buckets, expected_buckets):
    for key in buckets.keys():
        if not sum(buckets[key]) == expected_buckets[key]:
            return False
    return True


def test_create_buckets():
    # Test case 1: n = 10, num_in_buckets = [4, 0, 3]
    n = 10
    num_in_buckets = [4, 0, 3]
    buckets = create_buckets_one_query_per(n, num_in_buckets)
    expected_buckets = {
        "evid": num_in_buckets[0],
        "query": num_in_buckets[1],
        "unobs": num_in_buckets[2],
    }
    assert compare_buckets(buckets, expected_buckets), "Test case 1 failed"

    # Test case 2: n = 5, num_in_buckets = [2, 2, 1]
    n = 5
    num_in_buckets = [2, 2, 1]
    buckets = create_buckets_one_query_per(n, num_in_buckets)
    expected_buckets = {
        "evid": num_in_buckets[0],
        "query": num_in_buckets[1],
        "unobs": num_in_buckets[2],
    }
    assert compare_buckets(buckets, expected_buckets), "Test case 2 failed"

    # Test case 3: n = 8, num_in_buckets = [0, 0, 0]
    n = 8
    num_in_buckets = [0, 0, 0]
    buckets = create_buckets_one_query_per(n, num_in_buckets)
    expected_buckets = {
        "evid": num_in_buckets[0],
        "query": num_in_buckets[1],
        "unobs": num_in_buckets[2],
    }
    assert compare_buckets(buckets, expected_buckets), "Test case 3 failed"

    print("All test cases passed!")


# Run the test function
test_create_buckets()
