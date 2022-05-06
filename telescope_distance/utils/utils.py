from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import squareform


def pairwise_distance(series_list, distance_fn, condensed_form=True, pool_size=1):

    n = len(series_list)
    distance_vec = np.zeros(n * (n - 1) // 2)
    args = []
    idx_mapping = {}
    idx_counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            args.append((series_list[i], series_list[j],))
            idx_mapping[idx_counter] = n * i + j - ((i + 2) * (i + 1)) // 2
            idx_counter += 1
    pool = Pool(pool_size)
    distance_arr = pool.starmap(distance_fn, args)
    for i, val in enumerate(distance_arr):
        distance_vec[idx_mapping[i]] = val
    if condensed_form:
        return distance_vec
    return squareform(distance_vec, 'tomatrix')
