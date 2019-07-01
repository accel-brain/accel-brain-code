# -*- coding: utf-8 -*-
from pydbm.clustering.interface.computable_clustering_loss import ComputableClusteringLoss
from pydbm.loss.kl_divergence import KLDivergence
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class BalancedAssignmentsLoss(ComputableClusteringLoss):
    '''
    Balanced Assignments Loss.

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
        self.__kl_divergence = KLDivergence()
    
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
        if q_arr.shape[2] > 1:
            q_arr = np.nanmean(q_arr, axis=2)
        q_arr = q_arr.reshape((q_arr.shape[0], q_arr.shape[1]))

        cdef np.ndarray[DOUBLE_t, ndim=2] uniform_arr = np.random.uniform(
            low=q_arr.min(), 
            high=q_arr.max(), 
            size=q_arr.copy().shape
        )
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_ba_arr = self.__kl_divergence.compute_delta(
            q_arr,
            uniform_arr
        )
        delta_ba_arr = np.dot(feature_arr.T, delta_ba_arr).T
        delta_ba_arr = delta_ba_arr * self.__weight
        return (None, None, delta_ba_arr)
