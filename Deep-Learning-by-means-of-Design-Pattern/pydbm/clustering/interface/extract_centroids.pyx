# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ExtractableCentroids(metaclass=ABCMeta):
    '''
    The interface of clustering only to get information on 
    centroids to be mentioned as initial parameters in framework 
    of the Deep Embedded Clustering(DEC).

    References:
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''

    def extract_centroids(self, observed_arr, k):
        '''
        Clustering and extract centroids.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            k:                  The number of clusters.

        Returns:
            `np.ndarray` of centroids.
        '''
        raise NotImplementedError()
