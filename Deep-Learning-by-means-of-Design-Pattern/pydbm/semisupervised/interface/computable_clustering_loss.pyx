# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ComputableClusteringLoss(metaclass=ABCMeta):
    '''
    The interface of Loss functions in framework 
    of the Deep Embedded Clustering(DEC).

    References:
        - Aljalbout, E., Golkov, V., Siddiqui, Y., Strobel, M., & Cremers, D. (2018). Clustering with deep learning: Taxonomy and new methods. arXiv preprint arXiv:1801.07648.
        - Guo, X., Gao, L., Liu, X., & Yin, J. (2017, June). Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI (pp. 1753-1759).
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

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
        raise NotImplementedError()
