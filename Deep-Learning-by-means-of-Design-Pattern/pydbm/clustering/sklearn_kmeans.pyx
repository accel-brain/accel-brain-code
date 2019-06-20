# -*- coding: utf-8 -*-
from pydbm.clustering.interface.extractable_centroids import ExtractableCentroids
from sklearn.cluster import KMeans


class SklearnKMeans(ExtractableCentroids):
    '''
    K-Means method.

    The function of this class is only to get information on 
    centroids to be mentioned as initial parameters in framework 
    of the Deep Embedded Clustering(DEC).

    References:
        - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''
    
    def __init__(self, **kwargs):
        '''
        Init.

        Args:
            **kwargs:       Parameters of `sklearn.clustering.KMeans.__init__`.
        '''
        self.__kwargs = kwargs

    def extract_centroids(self, observed_arr, k):
        '''
        Clustering and extract centroids.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            k:                  The number of clusters.

        Returns:
            `np.ndarray` of centroids.
        '''
        if observed_arr.ndim != 2:
            observed_arr = observed_arr.reshape((observed_arr.shape[0], -1))
        return KMeans(n_clusters=k, **self.__kwargs).fit(observed_arr).cluster_centers_
