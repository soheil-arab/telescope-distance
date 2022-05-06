from scipy.cluster import hierarchy
from multiprocessing import Pool
import numpy as np
from telescope_distance.utils import utils


def agglomerative_clustering(metric, series_list=None, method='single', pool_size=1):
    """ Hierarchical clustering for time-series.

    Parameters
    ----------
    metric: callable, ndarray
        distance function to compute the pairwise distance between timeseries.
        Or a condensed distance ndarray of dissimilarities.

    series_list : list, ndarray, None
        list of ndarray of shape=(sz, d)
        ndarray of shape=(n_ts, sz, d)
        Time series dataset.
        If metric is given in the form of a distance matrix, series_list is no longer required

    method: str
        Which linkage criterion to use for agglomerative clustering.

    pool_size: int
        The size of multiprocessing pool to run distance matrix computations in parallel.

    Returns
    -------
    ndarray of shape (n_ts,)
        Cluster assignment
    """
    assert callable(metric) or isinstance(metric, np.ndarray), 'metric should be ndarray or a callable'

    distance_mat = utils.pairwise_distance(series_list, metric, True, pool_size) if callable(metric) else metric

    linkage_matrix = hierarchy.linkage(distance_mat, method)
    clustering_labels = hierarchy.fcluster(linkage_matrix, 2, criterion='maxclust')

    return clustering_labels


