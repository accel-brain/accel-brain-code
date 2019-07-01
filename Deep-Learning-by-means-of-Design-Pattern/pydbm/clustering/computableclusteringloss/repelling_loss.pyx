# -*- coding: utf-8 -*-
from pydbm.clustering.interface.computable_clustering_loss import ComputableClusteringLoss
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class RepellingLoss(ComputableClusteringLoss):
    '''
    Repelling Loss.

    Note that this class calculates this penalty term for each cluster 
    divided by soft assignments and refers to the sum as a regularizer.

    References:
        - Aljalbout, E., Golkov, V., Siddiqui, Y., Strobel, M., & Cremers, D. (2018). Clustering with deep learning: Taxonomy and new methods. arXiv preprint arXiv:1801.07648.
        - Guo, X., Gao, L., Liu, X., & Yin, J. (2017, June). Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI (pp. 1753-1759).
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.

    '''
    
    def __init__(self, weight=0.125):
        '''
        Init.

        Args:
            weight:     Weight of delta and loss.
        '''
        self.__weight = weight
    
    def compute_clustering_loss(
        self, 
        observed_arr, 
        reconstructed_arr, 
        feature_arr,
        delta_arr, 
        q_arr, 
        p_arr, 
    ):
        '''
        Compute clustering loss.

        Args:
            observed_arr:               `np.ndarray` of observed data points.
            reconstructed_arr:          `np.ndarray` of reconstructed data.
            feature_arr:                `np.ndarray` of feature points.
            delta_arr:                  `np.ndarray` of differences between feature points and centroids.
            p_arr:                      `np.ndarray` of result of soft assignment.
            q_arr:                      `np.ndarray` of target distribution.

        Returns:
            Tuple data.
            - `np.ndarray` of delta for the encoder.
            - `np.ndarray` of delta for the decoder.
            - `np.ndarray` of delta for the centroids.
        '''
        cdef np.ndarray label_arr = self.__assign_label(q_arr)
        cdef np.ndarray _feature_arr

        cdef int N = feature_arr.reshape((feature_arr.shape[0], -1)).shape[1]
        cdef int oN = observed_arr.reshape((observed_arr.shape[0], -1)).shape[1]
        cdef int s
        cdef np.ndarray pt_arr
        cdef np.ndarray penalty_arr = np.zeros(label_arr.shape[0])

        cdef np.ndarray[DOUBLE_t, ndim=2] penalty_delta_arr = np.zeros(feature_arr.copy().shape)
        cdef np.ndarray[DOUBLE_t, ndim=2] _penalty_delta_arr

        for label in label_arr:
            _feature_arr = feature_arr[label_arr == label]
            s = _feature_arr.shape[0]
            if s == 0:
                continue

            pt_arr = np.zeros(s ** 2)
            k = 0
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    pt_arr[k] = np.dot(
                        _feature_arr[i].T, 
                        _feature_arr[j]
                    ) / (
                        np.sqrt(
                            np.dot(
                                _feature_arr[i], 
                                _feature_arr[i]
                            )
                        ) * np.sqrt(
                            np.dot(
                                _feature_arr[j], 
                                _feature_arr[j]
                            )
                        )
                    )
                    k += 1

            penalty_term = np.nansum(pt_arr) / (N * (N - 1))
            _penalty_delta_arr = np.dot(
                penalty_term, 
                _feature_arr
            )
            penalty_delta_arr[label_arr == label] = _penalty_delta_arr

        penalty_delta_arr = penalty_delta_arr * self.__weight
        return (penalty_delta_arr, None, None)

    def __assign_label(self, q_arr):
        if q_arr.shape[2] > 1:
            q_arr = np.nanmean(q_arr, axis=2)
        q_arr = q_arr.reshape((q_arr.shape[0], q_arr.shape[1]))
        return q_arr.argmax(axis=1)
