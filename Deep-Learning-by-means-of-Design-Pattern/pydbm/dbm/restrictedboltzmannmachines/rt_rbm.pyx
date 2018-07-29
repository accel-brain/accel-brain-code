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
            observed_data_arr:    The `np.ndarray` of observed data points,
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
    
    def inference(
        self,
        np.ndarray[DOUBLE_t, ndim=3] observed_data_arr,
        int traning_count=-1,
        int r_batch_size=200,
        int training_count=1000
    ):
        '''
        Inferencing.
        
        Args:
            observed_data_arr:    The `np.ndarray` of observed data points,
                                  which is a rank-3 array-like or sparse matrix of shape: 
                                  (`The number of samples`, `The length of cycle`, `The number of features`)

            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.

                                  If you do not want to execute the mini-batch training, 
                                  the value of `batch_size` must be `-1`. 
                                  And `r_batch_size` is also parameter to control the mini-batch training 
                                  but is refered only in inference and reconstruction. 
                                  If this value is more than `0`, 
                                  the inferencing is a kind of reccursive learning with the mini-batch training.

        Returns:
            The `np.ndarray` of feature points.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.approximate_inferencing(
            observed_data_arr,
            training_count=training_count, 
            r_batch_size=r_batch_size
        )

        # The feature points can be observed data points.
        return self.graph.inferenced_arr

    def get_feature_points(self):
        '''
        Extract feature points from hidden layer.
        
        Returns:
            np.ndarray
        '''
        return self.graph.feature_points_arr
