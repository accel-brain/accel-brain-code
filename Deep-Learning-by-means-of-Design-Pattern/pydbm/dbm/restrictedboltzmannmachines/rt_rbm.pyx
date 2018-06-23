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
        cdef int j
        cdef int row_j
        cdef np.ndarray[DOUBLE_t, ndim=2] data_arr

        # Learning.
        for i in range(batch_size):
            data_arr = observed_data_arr[i:, :]
            row_j = data_arr.shape[0]
            for j in range(row_j):
                self.approximate_learning(
                    data_arr[j],
                    traning_count=traning_count, 
                    batch_size=batch_size
                )
    
    def inference(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
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
        cdef int j
        cdef int row_j
        cdef np.ndarray[DOUBLE_t, ndim=2] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] test_arr

        # Learning.
        result_arr_list = []
        for i in range(r_batch_size):
            data_arr = observed_data_arr[i:, :]
            row_j = data_arr.shape[0]
            for j in range(row_j):
                # Execute recursive learning.
                self.approximate_inferencing(
                    data_arr[j],
                    traning_count=traning_count, 
                    r_batch_size=r_batch_size
                )
                if j == row_j - 1:
                    # The feature points can be observed data points.
                    test_arr = self.graph.visible_activity_arr
                    result_arr_list.append(test_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] result_arr = np.array(result_arr_list)
        return result_arr
