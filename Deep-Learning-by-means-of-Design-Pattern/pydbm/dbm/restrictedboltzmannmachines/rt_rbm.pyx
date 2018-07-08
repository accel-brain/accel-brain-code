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
            observed_data_arr:    The `np.ndarray` of observed data points.
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
            observed_data_arr:    The `np.ndarray` of observed data points.
            r_batch_size:         Batch size.
        
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
