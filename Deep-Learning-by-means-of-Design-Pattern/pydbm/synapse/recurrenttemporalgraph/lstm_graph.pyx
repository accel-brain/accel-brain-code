# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse.recurrent_temporal_graph import RecurrentTemporalGraph
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.params_initializer import ParamsInitializer


class LSTMGraph(RecurrentTemporalGraph):
    '''
    Long short term memory(LSTM) networks
    based on Complete Bipartite Graph.

    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.

    In relation to do `transfer learning`, this object is-a `Synapse` which can be delegated to `LSTMModel`.
    '''
    
    ##
    # The parameters of LSTM.
    #
    #

    # Activation function to activate the observed data points in LSTM RNN layer.
    __observed_activating_function = None

    def get_observed_activating_function(self):
        ''' getter '''
        if isinstance(self.__observed_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __observed_activating_function must be `ActivatingFunctionInterface`.")
        return self.__observed_activating_function

    def set_observed_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __observed_activating_function must be `ActivatingFunctionInterface`.")
        self.__observed_activating_function = value

    observed_activating_function = property(get_observed_activating_function, set_observed_activating_function)

    # Activation function in LSTM input gate.
    __input_gate_activating_function = None
    
    def get_input_gate_activating_function(self):
        ''' getter '''
        if isinstance(self.__input_gate_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __visible_activating_function must be `ActivatingFunctionInterface`.")
        return self.__input_gate_activating_function

    def set_input_gate_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __visible_activating_function must be `ActivatingFunctionInterface`.")
        self.__input_gate_activating_function = value

    input_gate_activating_function = property(get_input_gate_activating_function, set_input_gate_activating_function)

    # Activation function in LSTM forget gate.
    __forget_gate_activating_function = None

    def get_forget_gate_activating_function(self):
        ''' getter '''
        if isinstance(self.__forget_gate_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __forget_gate_activating_function must be `ActivatingFunctionInterface`.")
        return self.__forget_gate_activating_function

    def set_forget_gate_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __forget_gate_activating_function must be `ActivatingFunctionInterface`.")
        self.__forget_gate_activating_function = value

    forget_gate_activating_function = property(get_forget_gate_activating_function, set_forget_gate_activating_function)

    # Activation function in LSTM output gate.
    __output_gate_activating_function = None

    def get_output_gate_activating_function(self):
        ''' getter '''
        if isinstance(self.__output_gate_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __output_gate_activating_function must be `ActivatingFunctionInterface`.")
        return self.__output_gate_activating_function

    def set_output_gate_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __output_gate_activating_function must be `ActivatingFunctionInterface`.")
        self.__output_gate_activating_function = value

    output_gate_activating_function = property(get_output_gate_activating_function, set_output_gate_activating_function)

    # Activation function in LSTM output layer.
    __output_activating_function = None

    def get_output_activating_function(self):
        ''' getter '''
        if isinstance(self.__output_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __output_activating_function must be `ActivatingFunctionInterface`.")
        return self.__output_activating_function

    def set_output_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __output_activating_function must be `ActivatingFunctionInterface`.")
        self.__output_activating_function = value

    output_activating_function = property(get_output_activating_function, set_output_activating_function)


    # Bias of neuron in LSTM RNN layer.
    __lstm_bias_arr = np.array([])

    def get_lstm_bias_arr(self):
        ''' getter '''
        if isinstance(self.__lstm_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __lstm_bias_arr must be `np.ndarray`.")

        return self.__lstm_bias_arr

    def set_lstm_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __lstm_bias_arr must be `np.ndarray`.")

        self.__lstm_bias_arr = value

    lstm_bias_arr = property(get_lstm_bias_arr, set_lstm_bias_arr)

    # Weight matrix which connects observed data points and hidden units in LSTM gates.
    __weights_lstm_observed_arr = np.array([])

    def get_weights_lstm_observed_arr(self):
        ''' getter '''
        if isinstance(self.__weights_lstm_observed_arr, np.ndarray) is False:
            raise TypeError("The type of __weights_lstm_observed_arr must be `np.ndarray`.")

        return self.__weights_lstm_observed_arr

    def set_weights_lstm_observed_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __weights_lstm_observed_arr must be `np.ndarray`.")

        self.__weights_lstm_observed_arr = value
    
    weights_lstm_observed_arr = property(get_weights_lstm_observed_arr, set_weights_lstm_observed_arr)

    # Weight matrix which connects hidden units as a remembered memory in LSTM gates.
    __weights_lstm_hidden_arr = np.array([])

    def get_weights_lstm_hidden_arr(self):
        ''' getter '''
        if isinstance(self.__weights_lstm_hidden_arr, np.ndarray) is False:
            raise TypeError("The type of __weights_lstm_hidden_arr must be `np.ndarray`.")

        return self.__weights_lstm_hidden_arr

    def set_weights_lstm_hidden_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __weights_lstm_hidden_arr must be `np.ndarray`.")

        self.__weights_lstm_hidden_arr = value
    
    weights_lstm_hidden_arr = property(get_weights_lstm_hidden_arr, set_weights_lstm_hidden_arr)

    # Weight matrix of the CEC for input gate.
    __weights_input_cec_arr = np.array([])

    def get_weights_input_cec_arr(self):
        ''' getter '''
        if isinstance(self.__weights_input_cec_arr, np.ndarray) is False:
            raise TypeError("The type of __weights_input_cec_arr must be `np.ndarray`.")

        return self.__weights_input_cec_arr

    def set_weights_input_cec_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __weights_input_cec_arr must be `np.ndarray`.")

        self.__weights_input_cec_arr = value
    
    weights_input_cec_arr = property(get_weights_input_cec_arr, set_weights_input_cec_arr)

    # Weight matrix of the CEC for forget gate.
    __weights_forget_cec_arr = np.array([])

    def get_weights_forget_cec_arr(self):
        ''' getter '''
        if isinstance(self.__weights_forget_cec_arr, np.ndarray) is False:
            raise TypeError("The type of __weights_forget_cec_arr must be `np.ndarray`.")

        return self.__weights_forget_cec_arr

    def set_weights_forget_cec_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __weights_forget_cec_arr must be `np.ndarray`.")

        self.__weights_forget_cec_arr = value
    
    weights_forget_cec_arr = property(get_weights_forget_cec_arr, set_weights_forget_cec_arr)

    # Weight matrix of the CEC for output gate.
    __weights_output_cec_arr = np.array([])

    def get_weights_output_cec_arr(self):
        ''' getter '''
        if isinstance(self.__weights_output_cec_arr, np.ndarray) is False:
            raise TypeError("The type of __weights_output_cec_arr must be `np.ndarray`.")

        return self.__weights_output_cec_arr

    def set_weights_output_cec_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __weights_output_cec_arr must be `np.ndarray`.")

        self.__weights_output_cec_arr = value
    
    weights_output_cec_arr = property(get_weights_output_cec_arr, set_weights_output_cec_arr)


    # Bias of neuron in output layer.
    __output_bias_arr = np.array([])

    def get_output_bias_arr(self):
        ''' getter '''
        if isinstance(self.__output_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __output_bias_arr must be `np.ndarray`.")

        return self.__output_bias_arr

    def set_output_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __output_bias_arr must be `np.ndarray`.")

        self.__output_bias_arr = value

    output_bias_arr = property(get_output_bias_arr, set_output_bias_arr)
    
    # Weight matrix of LSTM output layer.
    __weights_output_arr = np.array([])
    
    def get_weights_output_arr(self):
        ''' getter '''
        if isinstance(self.__weights_output_arr, np.ndarray) is False:
            raise TypeError("The type of __weights_output_arr must be `np.ndarray`.")

        return self.__weights_output_arr
    
    def set_weights_output_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __weights_output_arr must be `np.ndarray`.")

        self.__weights_output_arr = value

    weights_output_arr = property(get_weights_output_arr, set_weights_output_arr)


    ##
    # The parameters of LSTM-RTRBM.
    #
    #

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

    __diff_rbm_hidden_weights_arr = np.array([])
    
    def get_diff_rbm_hidden_weights_arr(self):
        ''' getter '''
        if isinstance(self.__diff_rbm_hidden_weights_arr, np.ndarray) is False:
            raise TypeError("The type of __diff_rbm_hidden_weights_arr must be `np.ndarray`.")
        return self.__diff_rbm_hidden_weights_arr

    def set_diff_rbm_hidden_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __diff_rbm_hidden_weights_arr must be `np.ndarray`.")
        self.__diff_rbm_hidden_weights_arr = value
    
    diff_rbm_hidden_weights_arr = property(get_diff_rbm_hidden_weights_arr, set_diff_rbm_hidden_weights_arr)

    __diff_rnn_visible_bias_arr = np.array([])
    
    def get_diff_rnn_visible_bias_arr(self):
        ''' getter '''
        if isinstance(self.__diff_rnn_visible_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __diff_rnn_visible_bias_arr must be `np.ndarray`.")
        return self.__diff_rnn_visible_bias_arr

    def set_diff_rnn_visible_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __diff_rnn_visible_bias_arr must be `np.ndarray`.")
        
        self.__diff_rnn_visible_bias_arr = value
    
    diff_rnn_visible_bias_arr = property(get_diff_rnn_visible_bias_arr, set_diff_rnn_visible_bias_arr)

    __diff_rnn_hidden_bias_arr = np.array([])
    
    def get_diff_rnn_hidden_bias_arr(self):
        ''' getter '''
        if isinstance(self.__diff_rnn_hidden_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __diff_rnn_hidden_bias_arr must be `np.ndarray`.")
        return self.__diff_rnn_hidden_bias_arr

    def set_diff_rnn_hidden_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __diff_rnn_hidden_bias_arr must be `np.ndarray`.")
        self.__diff_rnn_hidden_bias_arr = value
    
    diff_rnn_hidden_bias_arr = property(get_diff_rnn_hidden_bias_arr, set_diff_rnn_hidden_bias_arr)

    # The list of activities in RBM at step t-1.
    pre_hidden_activity_arr_list = []
    # delta of bias in RBM hidden layer.
    diff_hidden_bias_arr_list = []
    # delta of weight matrix in RNN hidden layer.
    diff_rnn_hidden_weights_arr_list = []

    def create_rnn_cells(
        self,
        int input_neuron_count=1,
        int hidden_neuron_count=1,
        int output_neuron_count=1,
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

        self.hidden_activity_arr = np.array([])
        self.cec_activity_arr = np.array([])

        self.weights_lstm_observed_arr = params_initializer.sample(
            size=(input_neuron_count, hidden_neuron_count * 4),
            **params_dict
        ) * scale

        self.weights_lstm_hidden_arr = params_initializer.sample(
            size=(hidden_neuron_count, hidden_neuron_count * 4),
            **params_dict
        ) * scale

        self.weights_input_cec_arr = params_initializer.sample(
            size=(hidden_neuron_count, hidden_neuron_count),
            **params_dict
        ) * scale

        self.weights_forget_cec_arr = params_initializer.sample(
            size=(hidden_neuron_count, hidden_neuron_count),
            **params_dict
        ) * scale

        self.weights_output_cec_arr = params_initializer.sample(
            size=(hidden_neuron_count, hidden_neuron_count),
            **params_dict
        ) * scale

        self.lstm_bias_arr = np.zeros(hidden_neuron_count * 4)

        self.weights_output_arr = params_initializer.sample(
            size=(hidden_neuron_count, output_neuron_count),
            **params_dict
        ) * scale

        self.output_bias_arr = np.zeros(output_neuron_count)

    def create_node(
        self,
        int shallower_neuron_count,
        int deeper_neuron_count,
        shallower_activating_function,
        deeper_activating_function,
        np.ndarray weights_arr=np.array([]),
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
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
                                                If this array is not empty, `ParamsInitializer.sample_f` will not be called 
                                                and `weights_arr` will be refered as initial weights.

            scale:                              Scale of parameters which will be `ParamsInitializer`.
            params_initializer:                 is-a `ParamsInitializer`.
            params_dict:                        `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.
        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        self.v_hat_weights_arr = params_initializer.sample(
            size=(shallower_neuron_count, deeper_neuron_count),
            **params_dict
        ) * scale
        self.hat_weights_arr = params_initializer.sample(
            size=(deeper_neuron_count, deeper_neuron_count),
            **params_dict
        ) * scale
        self.rbm_hidden_weights_arr = params_initializer.sample(
            size=(deeper_neuron_count, deeper_neuron_count),
            **params_dict
        ) * scale
        self.rnn_hidden_bias_arr = np.zeros((deeper_neuron_count, ))

        self.diff_rnn_hidden_weights_arr = np.zeros(
            (deeper_neuron_count, deeper_neuron_count)
        )
        
        self.diff_rbm_hidden_weights_arr = np.zeros(
            (deeper_neuron_count, deeper_neuron_count)
        )
        
        self.diff_rnn_visible_bias_arr = np.zeros((shallower_neuron_count, ))
        self.diff_rnn_hidden_bias_arr = np.zeros((deeper_neuron_count, ))

        super().create_node(
            shallower_neuron_count,
            deeper_neuron_count,
            shallower_activating_function,
            deeper_activating_function,
            weights_arr,
            scale,
            params_initializer,
            params_dict
        )
