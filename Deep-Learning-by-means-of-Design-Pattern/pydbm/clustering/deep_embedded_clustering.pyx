# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod
ctypedef np.float64_t DOUBLE_t
from pydbm.loss.kl_divergence import KLDivergence
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.clustering.interface.extract_centroids import ExtractableCentroids


class DeepEmbeddedClustering(metaclass=ABCMeta):
    '''
    Abstract class of the Deep Embedded Clustering(DEC).

    References:
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''
    # Degrees of freedom of the Student's t-distribution.
    __alpha = 1

    def get_alpha(self):
        ''' getter for degrees of freedom of the Student's t-distribution. '''
        return self.__alpha
    
    def set_alpha(self, value):
        ''' setter for degrees of freedom of the Student's t-distribution. '''
        self.__alpha = value

    alpha = property(get_alpha, set_alpha)

    # KL Divergence.
    __kl_divergence = KLDivergence()

    @abstractmethod
    def pre_learn(self, np.ndarray observed_arr):
        '''
        Pre-learning.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        '''
        raise NotImplementedError()

    @abstractmethod
    def embed_feature_points(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of feature points.
        '''
        raise NotImplementedError()

    def setup_initial_centroids(self, extractable_centroids, np.ndarray observed_arr, k=10):
        '''
        Create initial centroids and set it as property.

        Args:
            extractable_centroids:      is-a `ExtractableCentroids`.
            observed_arr:               `np.ndarray` of observed data points.
            k:                          The number of centroids.
        
        Returns:
            `np.ndarray` of centroids.
        '''
        if isinstance(extractable_centroids, ExtractableCentroids) is False:
            raise TypeError("The type `extract_centroids` must be `ExtractableCentroids`.")

        self.__mu_arr = extractable_centroids.extract_centroids(observed_arr, k)

    def learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
        '''
        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=2] train_observed_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] test_observed_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=2] batch_observed_arr

        if self.__test_size_rate > 0:
            train_index = np.random.choice(observed_arr.shape[0], round(self.__test_size_rate * observed_arr.shape[0]), replace=False)
            test_index = np.array(list(set(range(observed_arr.shape[0])) - set(train_index)))
            train_observed_arr = observed_arr[train_index]
            test_observed_arr = observed_arr[test_index]
        else:
            train_observed_arr = observed_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] test_pred_arr

        try:
            loss_list = []
            for epoch in range(self.__epochs):
                self.opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate / self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    loss = self.compute_loss(
                        pred_arr,
                        self.compute_target_distribution(pred_arr)
                    )
                    self.back_propagation()
                    self.optimize(learning_rate, epoch)

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        break
                    else:
                        raise

                if self.__test_size_rate > 0:
                    self.opt_params.inferencing_mode = True

                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]

                    test_pred_arr = self.inference(
                        test_batch_observed_arr
                    )
                    test_loss = self.compute_loss(
                        test_pred_arr,
                        self.compute_target_distribution(test_pred_arr)
                    )

                loss_list.append(loss)
                self.__logger.debug("Epoch: " + str(epoch) + " Train loss(KLD): " + str(loss) + " Test loss(KLD): " + str(test_loss))

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

    def inference(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.

        Returns:
            Predicted array like or sparse matrix.
        '''
        cdef np.ndarray pred_arr = self.forward_propagation(
            observed_arr
        )
        return pred_arr

    def forward_propagation(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points and do soft assignment.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of result of soft assignment.
        '''
        cdef np.ndarray feature_arr = self.embed_feature_points(observed_arr)
        cdef int batch_size = feature_arr.shape[0]
        if feature_arr.ndim != 2:
            default_shape = feature_arr.copy().shape
            feature_arr = feature_arr.reshape((batch_size, -1))
        else:
            default_shape = None

        cdef int k = self.__mu_arr.shape[0]
        cdef int dim = feature_arr.shape[1]
        cdef np.ndarray[DOUBLE_t, ndim=3] q_arr = np.zeros((
            batch_size, 
            k, 
            dim
        ))
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_arr = np.zeros((
            batch_size, 
            k, 
            dim
        ))

        for i in range(k):
            delta_arr[:, k] = feature_arr - self.__mu_arr[k]
            q_arr[:, k] = np.power((1 + np.square(delta_arr[:, k]) / self.__alpha), -(self.__alpha + 1) / 2)
        q_arr = q_arr / np.nansum(q_arr, axis=1)

        self.__delta_arr = delta_arr
        return q_arr

    def compute_target_distribution(self, q_arr):
        '''
        Compute target distribution.

        Args:
            q_arr:            `np.ndarray` of result of soft assignment.

        Returns:
            `np.ndarray` of target distribution.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] f_arr = np.nansum(q_arr, axis=2)
        cdef np.ndarray[DOUBLE_t, ndim=3] p_arr = np.power(q_arr, 2) / f_arr
        p_arr = p_arr / np.nansum(p_arr, axis=1)
        return p_arr
    
    def compute_loss(self, p_arr, q_arr):
        '''
        Compute loss.

        Args:
            p_arr:      `np.ndarray` of result of soft assignment.
            q_arr:      `np.ndarray` of target distribution.
        
        Returns:
            (loss, `np.ndarray` of delta)
        '''
        loss = self.__kl_divergence.compute_loss(p_arr, q_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_z_arr = ((self.__alpha + 1) / self.__alpha) * np.nansum(
            np.power(1 + self.__delta_arr / self.__alpha, -1) * (p_arr - q_arr) * (self.__delta_arr), 
            axis=1
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_mu_arr = -((self.__alpha + 1) / self.__alpha) * np.nansum(
            np.power(1 + self.__delta_arr / self.__alpha, -1)  * (p_arr - q_arr) * (self.__delta_arr),
            axis=2
        )

        self.__delta_z_arr = delta_z_arr
        self.__delta_mu_arr = delta_mu_arr

        return loss

    def back_propagation(self):
        '''
        Back propagation.

        Returns:
            `np.ndarray` of delta.
        '''
        self.backward_auto_encoder(self.__delta_z_arr)

    @abstractmethod
    def backward_auto_encoder(self, np.ndarray delta_arr):
        '''
        Pass down to the Auto-Encoder as backward.

        Args:
            delta_arr:      `np.ndarray` of delta.
        
        Returns:
            `np.ndarray` of delta.
        '''
        raise NotImplementedError()

    def optimize(self, learning_rate, epoch):
        '''
        Optimize.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        params_list = [
            self.__mu_arr
        ]
        grads_list = [
            self.__delta_mu_arr
        ]
        params_list = self.opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        self.__mu_arr = params_list[0]
        self.optimize_auto_encoder(learning_rate, epoch)

    @abstractmethod
    def optimize_auto_encoder(self, learning_rate, epoch):
        '''
        Optimize Auto-Encoder.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        raise NotImplementedError()
