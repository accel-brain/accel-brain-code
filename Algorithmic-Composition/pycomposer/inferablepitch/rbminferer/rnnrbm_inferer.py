# -*- coding: utf-8 -*-
import numpy as np
from pycomposer.inferablepitch.rbm_inferer import RBMInferer
from pydbm.dbm.builders.rnn_rbm_simple_builder import RNNRBMSimpleBuilder
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
from pydbm.approximation.rnnrbmcd.rnn_rbm_cd import RNNRBMCD
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.softmax_function import SoftmaxFunction


class RNNRBMInferer(RBMInferer):
    '''
    Inferacing next pitch by RNNRBM.
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

        # `Builder` in `Builder Pattern` for rnnrbm.
        rnnrbm_builder = RNNRBMSimpleBuilder()
        # Learning rate.
        rnnrbm_builder.learning_rate = learning_rate
        # Set units in visible layer.
        rnnrbm_builder.visible_neuron_part(
            SoftmaxFunction(), 
            127
        )
        # Set units in hidden layer.
        rnnrbm_builder.hidden_neuron_part(
            LogisticFunction(normalize_flag=False, binary_flag=hidden_binary_flag), 
            hidden_n
        )
        # Set units in RNN layer.
        rnnrbm_builder.rnn_neuron_part(
            LogisticFunction(normalize_flag=False, binary_flag=False)
        )
        # Set graph and approximation function.
        rnnrbm_builder.graph_part(RNNRBMCD())
        # Building.
        self.pitch_rbm = rnnrbm_builder.get_result()
