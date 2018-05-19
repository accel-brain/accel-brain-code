# -*- coding: utf-8 -*-
import numpy as np
from pycomposer.inferablepitch.rbm_pitch_inferer import RBMPitchInferer
from pydbm.dbm.builders.rt_rbm_simple_builder import RTRBMSimpleBuilder
from pydbm.approximation.rt_rbm_cd import RTRBMCD
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.softmax_function import SoftmaxFunction


class RTRBMPitchInferer(RBMPitchInferer):
    '''
    Inferacing next pitch by RTRBM.
    '''
    
    def __init__(
        self,
        learning_rate=0.00001,
        hidden_n=100,
        hidden_binary_flag=True,
        inferancing_training_count=1,
        r_batch_size=200
    ):
        self.inferancing_training_count = inferancing_training_count
        self.r_batch_size = r_batch_size

        # `Builder` in `Builder Pattern` for RTRBM.
        rtrbm_builder = RTRBMSimpleBuilder()
        # Learning rate.
        rtrbm_builder.learning_rate = learning_rate
        # Set units in visible layer.
        rtrbm_builder.visible_neuron_part(
            SoftmaxFunction(), 
            127
        )
        # Set units in hidden layer.
        rtrbm_builder.hidden_neuron_part(
            LogisticFunction(normalize_flag=False, binary_flag=hidden_binary_flag), 
            hidden_n
        )
        # Set units in RNN layer.
        rtrbm_builder.rnn_neuron_part(
            LogisticFunction(normalize_flag=False, binary_flag=False)
        )
        # Set graph and approximation function.
        rtrbm_builder.graph_part(RTRBMCD())
        # Building.
        self.rbm = rtrbm_builder.get_result()
