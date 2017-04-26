import math


def compute_distance(map1, map2):
    """
    Computes Euclidean distance between two sparse vectors, where the sparse vectors are
    represented as dicts with float values.
    @param map1: first sparse vector
    @param map2: second sparse vector
    """
    dist = 0.0
    union = set(map1.keys()).union(map2.keys())
    for key in union:
        map1val = map1.get(key, 0.0)
        map2val = map2.get(key, 0.0)
        dist += math.pow(map1val - map2val, 2)
    return math.sqrt(dist)
