# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.rtrbm_director import RTRBMDirector
from pydbm.dbm.builders.rt_rbm_simple_builder import RTRBMSimpleBuilder
from pydbm.approximation.rt_rbm_cd import RTRBMCD
from pydbm.optimization.opt_params import OptParams


class RecurrentTemporalRBM(object):
    '''
    The `Client` in Builder Pattern for building RTRBM.
    '''

    # Restricted Boltzmann Machine.
    __rbm = None
    
    def get_rbm(self):
        ''' getter '''
        return self.__rbm

    def set_rbm(self, value):
        ''' setter '''
        self.__rbm = value
    
    rbm = property(get_rbm, set_rbm)

    def __init__(
        self,
        visible_num,
        hidden_num,
        visible_activating_function,
        hidden_activating_function,
        rnn_activating_function,
        opt_params,
        learning_rate=1e-05
    ):
        '''
        Init.

        Args:
            visible_num:                    The number of units in visible layer.
            hidden_num:                     The number of units in hidden layer.
            visible_activating_function:    The activation function in visible layer.
            hidden_activating_function:     The activation function in hidden layer.
            opt_params:                     is-a `OptParams`.
            learning_rate:                  Learning rate.

        '''
        if isinstance(opt_params, OptParams) is False:
            raise TypeError()

        rtrbm_director = RTRBMDirector(RTRBMSimpleBuilder())
        rtrbm_director.rtrbm_construct(
            visible_num,
            hidden_num,
            visible_activating_function,
            hidden_activating_function,
            rnn_activating_function,
            RTRBMCD(
                opt_params=opt_params
            ),
            learning_rate=learning_rate
        )
        self.rbm = rtrbm_director.rbm

    def learn(self, observed_arr, training_count=1000, batch_size=200):
        '''
        Learning.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            training_count: The number of training.
            batch_size:     Batch size.
        '''
        # Learning.
        self.rbm.learn(
            # The `np.ndarray` of observed data points.
            observed_arr,
            # Training count.
            training_count=training_count, 
            # Batch size.
            batch_size=batch_size
        )

    def inference(self, test_arr, training_count=1, r_batch_size=-1):
        '''
        Inferencing and recursive learning.
        
        Args:
            test_arr:           `np.ndarray` of test data points.
            training_count:     The number of training.
            r_batch_size:       Batch size.

        Returns:
            `np.ndarray` of inferenced result.
        '''
        # Execute recursive learning.
        inferenced_arr = self.rbm.inference(
            test_arr,
            training_count=training_count, 
            r_batch_size=r_batch_size
        )
        return inferenced_arr

    def save_pre_learn_params(self, file_path):
        '''
        Save pre-learned parameters.
        
        Args:
            file_path:  Stored file path.
        '''
        self.rbm.graph.save_pre_learned_params(file_path)
