# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse.recurrent_temporal_graph import RecurrentTemporalGraph
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class LSTMGraph(RecurrentTemporalGraph):
    '''
    Long short term memory(LSTM) networks
    based on Complete Bipartite Graph.

    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.

    In relation to do `transfer learning`, this object is-a `Synapse` which can be delegated to `LSTMModel`.
    '''
    
    # Bias of neuron in RBM's hidden layer hidden layer.
    __hidden_bias_arr = np.array([])

    def get_hidden_bias_arr(self):
        ''' getter '''
        if isinstance(self.__hidden_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __hidden_bias_arr must be `np.ndarray`.")

        return self.__hidden_bias_arr

    def set_hidden_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __hidden_bias_arr must be `np.ndarray`.")

        self.__hidden_bias_arr = value

    hidden_bias_arr = property(get_hidden_bias_arr, set_hidden_bias_arr)

    # Activation function in RBM's hidden layer.
    def get_hidden_activating_function(self):
        ''' getter '''
        if isinstance(self.deeper_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __hidden_activating_function must be `ActivatingFunctionInterface`.")
        return self.deeper_activating_function

    def set_hidden_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __hidden_activating_function must be `ActivatingFunctionInterface`.")
        self.deeper_activating_function = value

    hidden_activating_function = property(get_hidden_activating_function, set_hidden_activating_function)


    # Activity of neuron in hidden layer in RBM.
    __hidden_activity_arr = np.array([])

    def get_hidden_activity_arr(self):
        ''' getter '''
        if isinstance(self.__hidden_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __hidden_activity_arr must be `np.ndarray`.")
        return self.__hidden_activity_arr

    def set_hidden_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __hidden_activity_arr must be `np.ndarray`.")
        self.__hidden_activity_arr = value

    hidden_activity_arr = property(get_hidden_activity_arr, set_hidden_activity_arr)

    # The list of `__hidden_bias_arr` to be memorized in RBM.
    __hidden_bias_arr_list = []
    
    def get_hidden_bias_arr_list(self):
        ''' getter '''
        return self.__hidden_bias_arr_list

    def set_hidden_bias_arr_list(self, value):
        ''' setter '''
        self.__hidden_bias_arr_list = value
    
    hidden_bias_arr_list = property(get_hidden_bias_arr_list, set_hidden_bias_arr_list)

    # The list of activities in RBM hidden layer.
    __rbm_hidden_activity_arr_list = []
    
    def get_rbm_hidden_activity_arr_list(self):
        ''' getter '''
        if isinstance(self.__rbm_hidden_activity_arr_list, list) is False:
            raise TypeError("The type of __rbm_hidden_activity_arr_list must be list.")
        return self.__rbm_hidden_activity_arr_list

    def set_rbm_hidden_activity_arr_list(self, value):
        ''' setter '''
        if isinstance(value, list) is False:
            raise TypeError("The type of __rbm_hidden_activity_arr_list must be list.")
        self.__rbm_hidden_activity_arr_list = value

    rbm_hidden_activity_arr_list = property(get_rbm_hidden_activity_arr_list, set_rbm_hidden_activity_arr_list)

    # The activity in RBM hidden layer at step t-1.
    __pre_rbm_hidden_activity_arr = np.array([])
    
    def get_pre_rbm_hidden_activity_arr(self):
        ''' getter '''
        if isinstance(self.__pre_rbm_hidden_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __pre_rbm_hidden_activity_arr must be `np.ndarray`.")
        return self.__pre_rbm_hidden_activity_arr

    def set_pre_rbm_hidden_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __pre_rbm_hidden_activity_arr must be `np.ndarray`.")
        self.__pre_rbm_hidden_activity_arr = value

    pre_rbm_hidden_activity_arr = property(get_pre_rbm_hidden_activity_arr, set_pre_rbm_hidden_activity_arr)

    # $W_{\hat{v}}$
    __v_hat_weights_arr = np.array([])

    def get_v_hat_weights_arr(self):
        ''' getter '''
        if isinstance(self.__v_hat_weights_arr, np.ndarray) is False:
            raise TypeError()
        return self.__v_hat_weights_arr

    def set_v_hat_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__v_hat_weights_arr = value
    
    v_hat_weights_arr = property(get_v_hat_weights_arr, set_v_hat_weights_arr)

    # $W_{\hat{h}}$
    __hat_weights_arr = np.array([])
    
    def get_hat_weights_arr(self):
        ''' getter '''
        if isinstance(self.__hat_weights_arr, np.ndarray) is False:
            raise TypeError()
        return self.__hat_weights_arr

    def set_hat_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__hat_weights_arr = value

    hat_weights_arr = property(get_hat_weights_arr, set_hat_weights_arr)

    __rnn_hidden_bias_arr = np.array([])
    
    def get_rnn_hidden_bias_arr(self):
        ''' getter '''
        if isinstance(self.__rnn_hidden_bias_arr, np.ndarray) is False:
            raise TypeError()
        return self.__rnn_hidden_bias_arr

    def set_rnn_hidden_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__rnn_hidden_bias_arr = value
    
    # Bias in RBM RNN layer.
    rnn_hidden_bias_arr = property(get_rnn_hidden_bias_arr, set_rnn_hidden_bias_arr)

    __visible_bias_arr_list = []
    
    def get_visible_bias_arr_list(self):
        ''' getter '''
        return self.__visible_bias_arr_list

    def set_visible_bias_arr_list(self, value):
        ''' setter '''
        self.__visible_bias_arr_list = value

    visible_bias_arr_list = property(get_visible_bias_arr_list, set_visible_bias_arr_list)
    
    # The list of activities in RBM at step t-1.
    __pre_hidden_activity_arr_list = []
    
    def get_pre_hidden_activity_arr_list(self):
        ''' getter '''
        if isinstance(self.__pre_hidden_activity_arr_list, list) is False:
            raise TypeError()
        return self.__pre_hidden_activity_arr_list

    def set_pre_hidden_activity_arr_list(self, value):
        ''' setter '''
        if isinstance(value, list) is False:
            raise TypeError()
        self.__pre_hidden_activity_arr_list = value

    pre_hidden_activity_arr_list = property(get_pre_hidden_activity_arr_list, set_pre_hidden_activity_arr_list)

    # delta of bias in RBM visible layer.
    __diff_visible_bias_arr_list = []
    
    def get_diff_visible_bias_arr_list(self):
        ''' getter '''
        return self.__diff_visible_bias_arr_list

    def set_diff_visible_bias_arr_list(self, value):
        ''' setter '''
        self.__diff_visible_bias_arr_list = value
    
    diff_visible_bias_arr_list = property(get_diff_visible_bias_arr_list, set_diff_visible_bias_arr_list)

    # delta of bias in RBM hidden layer.
    __diff_hidden_bias_arr_list = []

    def get_diff_hidden_bias_arr_list(self):
        ''' getter '''
        return self.__diff_hidden_bias_arr_list

    def set_diff_hidden_bias_arr_list(self, value):
        ''' setter '''
        self.__diff_hidden_bias_arr_list = value
    
    diff_hidden_bias_arr_list = property(get_diff_hidden_bias_arr_list, set_diff_hidden_bias_arr_list)

    # Weight matrix in RBM hidden layer.
    __rbm_hidden_weights_arr = np.array([])
    
    def get_rbm_hidden_weights_arr(self):
        ''' getter '''
        if isinstance(self.__rbm_hidden_weights_arr, np.ndarray) is False:
            raise TypeError("The type of __rbm_hidden_weights_arr must be `np.ndarray`.")
        return self.__rbm_hidden_weights_arr

    def set_rbm_hidden_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __rbm_hidden_weights_arr must be `np.ndarray`.")
        self.__rbm_hidden_weights_arr = value
    
    rbm_hidden_weights_arr = property(get_rbm_hidden_weights_arr, set_rbm_hidden_weights_arr)

    def create_rnn_cells(
        self,
        int input_neuron_count,
        int hidden_neuron_count,
        int output_neuron_count
    ):
        self.hidden_activity_arr = np.zeros((hidden_neuron_count, ))
        self.rnn_activity_arr = np.zeros((hidden_neuron_count, ))

        self.weights_given_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.001
        self.weights_input_gate_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.001
        self.weights_forget_gate_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.001
        self.weights_output_gate_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.001
        
        self.weights_hidden_arr = np.random.normal(size=(hidden_neuron_count, hidden_neuron_count)) * 0.001

        self.weights_output_arr = np.random.normal(size=(hidden_neuron_count, output_neuron_count)) * 0.001

        self.given_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.001
        self.input_gate_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.001
        self.forget_gate_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.001
        self.output_gate_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.001
        self.hidden_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.001

        self.output_bias_arr = np.random.normal(size=output_neuron_count) * 0.001

    def create_node(
        self,
        int shallower_neuron_count,
        int deeper_neuron_count,
        shallower_activating_function,
        deeper_activating_function,
        np.ndarray weights_arr=np.array([])
    ):
        '''
        Set links of nodes to the graphs.

        Override.

        Args:
            shallower_neuron_count:             The number of neurons in shallower layer.
            deeper_neuron_count:                The number of neurons in deeper layer.
            shallower_activating_function:      The activation function in shallower layer.
            deeper_activating_function:         The activation function in deeper layer.
            weights_arr:                        The weights of links.
        '''
        self.v_hat_weights_arr = np.zeros(
            (shallower_neuron_count, deeper_neuron_count)
        )
        self.hat_weights_arr = np.zeros(
            (deeper_neuron_count, deeper_neuron_count)
        )
        self.rbm_hidden_weights_arr = np.zeros(
            (deeper_neuron_count, deeper_neuron_count)
        )
        self.rnn_hidden_bias_arr = np.random.uniform(low=0, high=1, size=(deeper_neuron_count, ))

        super().create_node(
            shallower_neuron_count,
            deeper_neuron_count,
            shallower_activating_function,
            deeper_activating_function,
            weights_arr
        )
