# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pycomposer.inferableduration.rbm_time_diff_inferer import RBMTimeDiffInferer
from pydbm.dbm.builders.rnn_rbm_simple_builder import RNNRBMSimpleBuilder
from pydbm.approximation.rtrbmcd.rnn_rbm_cd import RNNRBMCD
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.softmax_function import SoftmaxFunction


class RNNRBMTimeDiffInferer(RBMTimeDiffInferer):
    '''
    Inferacing duration by RNNRBM.
    '''

    def __init__(
        self,
        tone_df,
        learning_rate=0.00001,
        beat_n=4,
        hidden_n=100,
        hidden_binary_flag=True,
        inferancing_training_count=1,
        r_batch_size=200
    ):
        self.inferancing_training_count = inferancing_training_count
        self.r_batch_size = r_batch_size
        self.tone_df = tone_df
        visible_n = self.tone_df.end.astype(int).max()

        # `Builder` in `Builder Pattern` for RTRBM.
        rnnrbm_builder = RNNRBMSimpleBuilder()
        # Learning rate.
        rnnrbm_builder.learning_rate = learning_rate
        # Set units in visible layer.
        rnnrbm_builder.visible_neuron_part(
            LogisticFunction(normalize_flag=True, binary_flag=False), 
            beat_n
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
        self.rbm = rnnrbm_builder.get_result()
