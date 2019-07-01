# -*- coding: utf-8 -*-
from pydbm.clustering.interface.computable_clustering_loss import ComputableClusteringLoss
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.loss.mean_squared_error import MeanSquaredError
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ReconstructionLoss(ComputableClusteringLoss):
    '''
    Reconstruction Loss.

    References:
        - Aljalbout, E., Golkov, V., Siddiqui, Y., Strobel, M., & Cremers, D. (2018). Clustering with deep learning: Taxonomy and new methods. arXiv preprint arXiv:1801.07648.
        - Guo, X., Gao, L., Liu, X., & Yin, J. (2017, June). Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI (pp. 1753-1759).
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.

    '''
    
    def __init__(self, weight=0.125, computable_loss=None):
        '''
        Init.

        Args:
            weight:     Weight of delta and loss.
        '''
        self.__weight = weight
        if computable_loss is None:
            self.__computable_loss = MeanSquaredError()
        else:
            if isinstance(computable_loss, ComputableLoss) is False:
                raise TypeError()
            self.__computable_loss = computable_loss
    
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
        cdef np.ndarray delta_rec_arr = self.__computable_loss.compute_delta(
            reconstructed_arr, 
            observed_arr
        )
        delta_rec_arr = delta_rec_arr * self.__weight
        return (None, delta_rec_arr, None)
