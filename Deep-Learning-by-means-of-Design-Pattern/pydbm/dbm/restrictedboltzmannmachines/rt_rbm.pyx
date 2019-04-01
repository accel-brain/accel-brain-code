# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
ctypedef np.float64_t DOUBLE_t


class RTRBM(RestrictedBoltzmannMachine):
    '''
    Reccurent temploral restricted boltzmann machine.
    '''

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=3] observed_data_arr,
        int traning_count=-1,
        int batch_size=200,
        int training_count=1000
    ):
        '''
        Learning.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points,
                                    which is a rank-3 array-like or sparse matrix of shape: 
                                    (`The number of samples`, `The length of cycle`, `The number of features`)

            traning_count:        Training counts.
            batch_size:           Batch size.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        # Learning.
        self.approximate_learning(
            observed_data_arr,
            training_count=training_count, 
            batch_size=batch_size
        )
        self.__batch_size = batch_size
    
    def inference(
        self,
        np.ndarray[DOUBLE_t, ndim=3] observed_data_arr,
        int traning_count=-1,
        int r_batch_size=200,
        int training_count=1000,
        batch_size=None
    ):
        '''
        Inferencing.
        
        Args:
            observed_data_arr:      The `np.ndarray` of observed data points,
                                    which is a rank-3 array-like or sparse matrix of shape: 
                                    (`The number of samples`, `The length of cycle`, `The number of features`)

            r_batch_size:           Batch size.
                                    If this value is `0`, the inferencing is a recursive learning.
                                    If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                    If this value is '-1', the inferencing is not a recursive learning.

                                    If you do not want to execute the mini-batch training, 
                                    the value of `batch_size` must be `-1`. 
                                    And `r_batch_size` is also parameter to control the mini-batch training 
                                    but is refered only in inference and reconstruction. 
                                    If this value is more than `0`, 
                                    the inferencing is a kind of reccursive learning with the mini-batch training.

            batch_size:             Batch size in learning.

        Returns:
            The `np.ndarray` of feature points.
        '''
        if batch_size is not None:
            self.__batch_size = batch_size

        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        feature_points_arr = None
        if self.__batch_size > 0:
            for i in range(int(observed_data_arr.shape[0] / self.__batch_size)):
                start_index = i * self.__batch_size
                end_index = (i + 1) * self.__batch_size

                self.approximate_inferencing(
                    observed_data_arr[start_index:end_index],
                    training_count=training_count, 
                    r_batch_size=r_batch_size
                )
                if feature_points_arr is None:
                    feature_points_arr = self.graph.inferenced_arr
                else:
                    feature_points_arr = np.r_[feature_points_arr, self.graph.inferenced_arr]
        else:
            self.approximate_inferencing(
                observed_data_arr,
                training_count=training_count, 
                r_batch_size=r_batch_size
            )
            feature_points_arr = self.graph.inferenced_arr

        return feature_points_arr

    def get_feature_points(self):
        '''
        Extract feature points from hidden layer.
        
        Returns:
            `np.ndarray` of feature points.
        '''
        return self.graph.hidden_activity_arr

    def get_reconstructed_arr(self):
        '''
        Extract reconstructed points.

        Returns:
            `np.ndarray` of reconstructed points.
        '''
        return self.graph.reconstructed_arr

    def get_reconstruct_error_arr(self):
        '''
        Extract reconstructed errors.

        Retruns:
            `np.ndarray` of reconstructed errors.
        '''
        return self.graph.reconstruct_error_arr
