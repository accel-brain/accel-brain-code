# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
from pydbm.dbm.restrictedboltzmannmachines.rt_rbm import RTRBM
ctypedef np.float64_t DOUBLE_t


class LSTMRTRBM(RTRBM):
    '''
    LSTM-RTRBM.
    '''

    def extract_transfered_params(
        self,
        int input_neuron_count,
        int hidden_neuron_count,
        int output_neuron_count
    ):
        '''
        Extract transfered parameters.
        
        Args:
            input_neuron_count:     The number of units in input layer.
            hidden_neuron_count:    The number of units in hidden layer.
            output_neuron_count:    The number of units in output layer.
        
        Returns:
            `Synapse` which has pre-learned parameters.
        '''
        self.graph.create_rnn_cells(
            input_neuron_count,
            hidden_neuron_count,
            output_neuron_count
        )
        
        if self.graph.v_hat_weights_arr.shape[0] != input_neuron_count:
            raise ValueError("The shape of pre-learned parameter must be the shape of transfer-learned parameters.")
        if self.graph.v_hat_weights_arr.shape[1] != hidden_neuron_count:
            raise ValueError("The shape of pre-learned parameter must be the shape of transfer-learned parameters.")
        if self.graph.rbm_hidden_weights_arr.shape[0] != hidden_neuron_count:
            raise ValueError("The shape of pre-learned parameter must be the shape of transfer-learned parameters.")
        if self.graph.rbm_hidden_weights_arr.shape[1] != hidden_neuron_count:
            raise ValueError("The shape of pre-learned parameter must be the shape of transfer-learned parameters.")

        self.graph.weights_xg_arr += self.graph.v_hat_weights_arr
        self.graph.weights_xi_arr += self.graph.v_hat_weights_arr
        self.graph.weights_xf_arr += self.graph.v_hat_weights_arr
        self.graph.weights_xo_arr += self.graph.v_hat_weights_arr
        
        self.graph.weights_hg_arr += self.graph.rbm_hidden_weights_arr
        self.graph.weights_hi_arr += self.graph.rbm_hidden_weights_arr
        self.graph.weights_hf_arr += self.graph.rbm_hidden_weights_arr
        self.graph.weights_ho_arr += self.graph.rbm_hidden_weights_arr

        return self.graph
