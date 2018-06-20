# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
ctypedef np.float64_t DOUBLE_t


class RTRBM(RestrictedBoltzmannMachine):
    '''
    Reccurent temploral restricted boltzmann machine.
    '''

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=1000,
        int batch_size=200
    ):
        '''
        Learning.

        Args:
            observed_data_arr:    The `np.ndarray` of observed data points.
            traning_count:        Training counts.
            batch_size:           Batch size.
        '''
        cdef int i
        cdef int row_i
        cdef int batch
        cdef np.ndarray[DOUBLE_t, ndim=1] data_arr

        # Learning.
        for batch in range(batch_size):
            row_i = observed_data_arr.shape[0] - batch
            for i in range(batch, row_i):
                data_arr = observed_data_arr[i]
                self.approximate_learning(
                    data_arr[i],
                    traning_count=traning_count, 
                    batch_size=batch_size
                )
    
    def inference(
        self,
        np.ndarray observed_data_arr,
        int traning_count=1000,
        int r_batch_size=200
    ):
        '''
        Inferencing.
        
        Args:
            observed_data_arr:    The `np.ndarray` of observed data points.
            r_batch_size:         Batch size.
        
        Returns:
            The `np.ndarray` of feature points.
        '''
        cdef int i
        cdef int row_i = observed_data_arr.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] test_arr = observed_data_arr[0]
        cdef np.ndarray[DOUBLE_t, ndim=2] result_arr = np.array([None] * observed_data_arr.shape[0])
        for i in range(row_i):
            # Execute recursive learning.
            self.approximate_inferencing(
                test_arr,
                traning_count=traning_count, 
                r_batch_size=r_batch_size
            )
            # The feature points can be observed data points.
            result_arr[i] = test_arr = self.graph.visible_activity_arr

        return result_arr
