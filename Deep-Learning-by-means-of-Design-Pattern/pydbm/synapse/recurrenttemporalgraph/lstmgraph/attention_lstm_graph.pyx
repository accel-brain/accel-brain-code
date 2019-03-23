# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph


class AttentionLSTMGraph(LSTMGraph):
    '''
    Attention based on Long short term memory(LSTM) networks.

    In relation to do `transfer learning`, this object is-a `Synapse` which can be delegated to `AttentionLSTMModel`.

    '''

    # $W_{\hat{v}}$
    __attention_output_weight_arr = np.array([])

    def get_attention_output_weight_arr(self):
        ''' getter '''
        if isinstance(self.__attention_output_weight_arr, np.ndarray) is False:
            raise TypeError()
        return self.__attention_output_weight_arr

    def set_attention_output_weight_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__attention_output_weight_arr = value
    
    attention_output_weight_arr = property(get_attention_output_weight_arr, set_attention_output_weight_arr)

    def create_rnn_cells(
        self,
        int input_neuron_count,
        int hidden_neuron_count,
        int output_neuron_count
    ):
        super().create_rnn_cells(
            input_neuron_count,
            hidden_neuron_count,
            output_neuron_count
        )
        self.attention_output_weight_arr = np.random.normal(size=(hidden_neuron_count*2, output_neuron_count)).astype(np.float16) * 0.1
