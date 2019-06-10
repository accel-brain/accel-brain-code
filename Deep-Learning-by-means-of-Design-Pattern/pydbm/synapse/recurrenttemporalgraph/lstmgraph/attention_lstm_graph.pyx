# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse.recurrenttemporalgraph.lstm_graph import LSTMGraph
from pydbm.params_initializer import ParamsInitializer


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
        int output_neuron_count,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Create RNN cells for a `LSTMModel`.

        Args:
            input_neuron_count:     The number of units in input layer.
            hidden_neuron_count:    The number of units in hidden layer.
            output_neuron_count:    The number of units in output layer.
            scale:                  Scale of parameters which will be `ParamsInitializer`.
            params_initializer:     is-a `ParamsInitializer`.
            params_dict:            `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.
        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        super().create_rnn_cells(
            input_neuron_count,
            hidden_neuron_count,
            output_neuron_count,
            scale,
            params_initializer,
            params_dict
        )
        self.attention_output_weight_arr = params_initializer.sample(
            size=(hidden_neuron_count*2, output_neuron_count),
            **params_dict
         ) * scale
