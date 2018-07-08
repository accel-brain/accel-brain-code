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

    # Activity of neuron in visible layer.
    __input_activity_arr = np.array([])

    def get_input_activity_arr(self):
        ''' getter '''
        if isinstance(self.__input_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __input_activity_arr must be `np.ndarray`.")

        return self.__input_activity_arr

    def set_input_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __input_activity_arr must be `np.ndarray`.")

        self.__input_activity_arr = value

    input_activity_arr = property(get_input_activity_arr, set_input_activity_arr)

    # Activity of neuron in hidden layer.
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
    
    __rnn_activity_arr = np.array([])
    
    def get_rnn_activity_arr(self):
        ''' getter '''
        if isinstance(self.__rnn_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __rnn_activity_arr must be `np.ndarray`.")
        return self.__rnn_activity_arr

    def set_rnn_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __rnn_activity_arr must be `np.ndarray`.")
        self.__rnn_activity_arr = value
    
    rnn_activity_arr = property(get_rnn_activity_arr, set_rnn_activity_arr)

    # Activity of neuron in hidden layer.
    __output_activity_arr = np.array([])

    def get_output_activity_arr(self):
        ''' getter '''
        if isinstance(self.__output_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __output_activity_arr must be `np.ndarray`.")
        return self.__output_activity_arr

    def set_output_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __output_activity_arr must be `np.ndarray`.")
        self.__output_activity_arr = value

    output_activity_arr = property(get_output_activity_arr, set_output_activity_arr)

    # Activity of neuron in hidden layer.
    __forget_activity_arr = np.array([])

    def get_forget_activity_arr(self):
        ''' getter '''
        if isinstance(self.__forget_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __forget_activity_arr must be `np.ndarray`.")
        return self.__forget_activity_arr

    def set_forget_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __forget_activity_arr must be `np.ndarray`.")
        self.__forget_activity_arr = value

    forget_activity_arr = property(get_forget_activity_arr, set_forget_activity_arr)

    # Activity of neuron in hidden layer.
    __linear_activity_arr = np.array([])

    def get_linear_activity_arr(self):
        ''' getter '''
        if isinstance(self.__linear_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __linear_activity_arr must be `np.ndarray`.")
        return self.__linear_activity_arr

    def set_linear_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __linear_activity_arr must be `np.ndarray`.")
        self.__linear_activity_arr = value

    linear_activity_arr = property(get_linear_activity_arr, set_linear_activity_arr)

    # Bias of neuron in visible layer.
    __input_bias_arr = np.array([])

    def get_input_bias_arr(self):
        ''' getter '''
        if isinstance(self.__input_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __input_bias_arr must be `np.ndarray`.")

        return self.__input_bias_arr

    def set_input_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __input_bias_arr must be `np.ndarray`.")

        self.__input_bias_arr = value

    input_bias_arr = property(get_input_bias_arr, set_input_bias_arr)

    # Bias of neuron in hidden layer.
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

    # Bias of neuron in visible layer.
    __rnn_output_bias_arr = np.array([])

    def get_rnn_output_bias_arr(self):
        ''' getter '''
        if isinstance(self.__rnn_output_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __rnn_output_bias_arr must be `np.ndarray`.")

        return self.__rnn_output_bias_arr

    def set_rnn_output_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __rnn_output_bias_arr must be `np.ndarray`.")

        self.__rnn_output_bias_arr = value

    rnn_output_bias_arr = property(get_rnn_output_bias_arr, set_rnn_output_bias_arr)

    # Bias of neuron in visible layer.
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

    # Bias of neuron in visible layer.
    __forget_bias_arr = np.array([])

    def get_forget_bias_arr(self):
        ''' getter '''
        if isinstance(self.__forget_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __forget_bias_arr must be `np.ndarray`.")

        return self.__forget_bias_arr

    def set_forget_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __forget_bias_arr must be `np.ndarray`.")

        self.__forget_bias_arr = value

    forget_bias_arr = property(get_forget_bias_arr, set_forget_bias_arr)

    # Bias of neuron in visible layer.
    __linear_bias_arr = np.array([])

    def get_linear_bias_arr(self):
        ''' getter '''
        if isinstance(self.__linear_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __linear_bias_arr must be `np.ndarray`.")

        return self.__linear_bias_arr

    def set_linear_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __linear_bias_arr must be `np.ndarray`.")

        self.__linear_bias_arr = value

    linear_bias_arr = property(get_linear_bias_arr, set_linear_bias_arr)

    __output_layer_bias_arr = np.array([])

    def get_output_layer_bias_arr(self):
        ''' getter '''
        if isinstance(self.__output_layer_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __output_layer_bias_arr must be `np.ndarray`.")
        return self.__output_layer_bias_arr
    
    def set_output_layer_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __output_layer_bias_arr must be `np.ndarray`.")
        self.__output_layer_bias_arr = value
    
    output_layer_bias_arr = property(get_output_layer_bias_arr, set_output_layer_bias_arr)

    __observed_activating_function = None

    # Activation function in visible layer.
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

    # Activation function in visible layer.
    def get_input_activating_function(self):
        ''' getter '''
        if isinstance(self.shallower_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __visible_activating_function must be `ActivatingFunctionInterface`.")
        return self.shallower_activating_function

    def set_input_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __visible_activating_function must be `ActivatingFunctionInterface`.")
        self.shallower_activating_function = value

    input_activating_function = property(get_input_activating_function, set_input_activating_function)

    # Activation function in hidden layer.
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

    __forget_activating_function = None

    # Activation function in hidden layer.
    def get_forget_activating_function(self):
        ''' getter '''
        if isinstance(self.__forget_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __forget_activating_function must be `ActivatingFunctionInterface`.")
        return self.__forget_activating_function

    def set_forget_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __forget_activating_function must be `ActivatingFunctionInterface`.")
        self.__forget_activating_function = value

    forget_activating_function = property(get_forget_activating_function, set_forget_activating_function)

    __rnn_output_activating_function = None

    # Activation function in hidden layer.
    def get_rnn_output_activating_function(self):
        ''' getter '''
        if isinstance(self.__rnn_output_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __rnn_output_activating_function must be `ActivatingFunctionInterface`.")
        return self.__rnn_output_activating_function

    def set_rnn_output_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __rnn_output_activating_function must be `ActivatingFunctionInterface`.")
        self.__rnn_output_activating_function = value

    rnn_output_activating_function = property(get_rnn_output_activating_function, set_rnn_output_activating_function)

    __output_activating_function = None

    # Activation function in hidden layer.
    def get_output_activating_function(self):
        ''' getter '''
        if isinstance(self.__output_activating_function, ActivatingFunctionInterface) is False and self.__output_activating_function is not None:
            raise TypeError("The type of __output_activating_function must be `ActivatingFunctionInterface`.")
        return self.__output_activating_function

    def set_output_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False and value is not None:
            raise TypeError("The type of __output_activating_function must be `ActivatingFunctionInterface`.")
        self.__output_activating_function = value

    output_activating_function = property(get_output_activating_function, set_output_activating_function)

    __linear_activating_function = None

    # Activation function in hidden layer.
    def get_linear_activating_function(self):
        ''' getter '''
        if isinstance(self.__linear_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __linear_activating_function must be `ActivatingFunctionInterface`.")
        return self.__linear_activating_function

    def set_linear_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __linear_activating_function must be `ActivatingFunctionInterface`.")
        self.__linear_activating_function = value

    linear_activating_function = property(get_linear_activating_function, set_linear_activating_function)
    
    __weights_xg_arr = np.array([])
    
    def get_weights_xg_arr(self):
        ''' getter '''
        return self.__weights_xg_arr

    def set_weights_xg_arr(self, value):
        ''' setter '''
        self.__weights_xg_arr = value
    
    weights_xg_arr = property(get_weights_xg_arr, set_weights_xg_arr)

    __weights_xi_arr = np.array([])
    
    def get_weights_xi_arr(self):
        ''' getter '''
        return self.__weights_xi_arr

    def set_weights_xi_arr(self, value):
        ''' setter '''
        self.__weights_xi_arr = value
    
    weights_xi_arr = property(get_weights_xi_arr, set_weights_xi_arr)
    
    __weights_xf_arr = np.array([])

    def get_weights_xf_arr(self):
        ''' getter '''
        return self.__weights_xf_arr

    def set_weights_xf_arr(self, value):
        ''' setter '''
        self.__weights_xf_arr = value
    
    weights_xf_arr = property(get_weights_xf_arr, set_weights_xf_arr)

    __weights_xo_arr = np.array([])

    def get_weights_xo_arr(self):
        ''' getter '''
        return self.__weights_xo_arr

    def set_weights_xo_arr(self, value):
        ''' setter '''
        self.__weights_xo_arr = value
    
    weights_xo_arr = property(get_weights_xo_arr, set_weights_xo_arr)

    __weights_hg_arr = np.array([])
    
    def get_weights_hg_arr(self):
        ''' getter '''
        return self.__weights_hg_arr

    def set_weights_hg_arr(self, value):
        ''' setter '''
        self.__weights_hg_arr = value
    
    weights_hg_arr = property(get_weights_hg_arr, set_weights_hg_arr)

    __weights_hi_arr = np.array([])
    
    def get_weights_hi_arr(self):
        ''' getter '''
        return self.__weights_hi_arr

    def set_weights_hi_arr(self, value):
        ''' getter '''
        self.__weights_hi_arr = value

    __weights_hi_arr = np.array([])

    def set_weights_hi_arr(self, value):
        ''' setter '''
        self.__weights_hi_arr = value

    weights_hi_arr = property(get_weights_hi_arr, set_weights_hi_arr)

    __weights_hf_arr = np.array([])

    def get_weights_hf_arr(self):
        ''' getter '''
        return self.__weights_hf_arr

    def set_weights_hf_arr(self, value):
        ''' setter '''
        self.__weights_hf_arr = value
    
    weights_hf_arr = property(get_weights_hf_arr, set_weights_hf_arr)

    __weights_hy_arr = np.array([])

    def get_weights_hy_arr(self):
        ''' getter '''
        return self.__weights_hy_arr

    def set_weights_hy_arr(self, value):
        ''' setter '''
        self.__weights_hy_arr = value
    
    weights_hy_arr = property(get_weights_hy_arr, set_weights_hy_arr)

    __weights_hidden_output_arr = np.array([])
    
    def get_weights_hidden_output_arr(self):
        ''' getter '''
        return self.__weights_hidden_output_arr

    def set_weights_hidden_output_arr(self, value):
        ''' setter '''
        self.__weights_hidden_output_arr = value
    
    weights_hidden_output_arr = property(get_weights_hidden_output_arr, set_weights_hidden_output_arr)

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
    
    rnn_hidden_bias_arr = property(get_rnn_hidden_bias_arr, set_rnn_hidden_bias_arr)

    __visible_bias_arr_list = []
    
    def get_visible_bias_arr_list(self):
        ''' getter '''
        return self.__visible_bias_arr_list

    def set_visible_bias_arr_list(self, value):
        ''' setter '''
        self.__visible_bias_arr_list = value

    visible_bias_arr_list = property(get_visible_bias_arr_list, set_visible_bias_arr_list)
    
    __hidden_bias_arr_list = []
    
    def get_hidden_bias_arr_list(self):
        ''' getter '''
        return self.__hidden_bias_arr_list

    def set_hidden_bias_arr_list(self, value):
        ''' setter '''
        self.__hidden_bias_arr_list = value
    
    hidden_bias_arr_list = property(get_hidden_bias_arr_list, set_hidden_bias_arr_list)

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

    __diff_visible_bias_arr_list = []
    
    def get_diff_visible_bias_arr_list(self):
        ''' getter '''
        return self.__diff_visible_bias_arr_list

    def set_diff_visible_bias_arr_list(self, value):
        ''' setter '''
        self.__diff_visible_bias_arr_list = value
    
    diff_visible_bias_arr_list = property(get_diff_visible_bias_arr_list, set_diff_visible_bias_arr_list)

    __diff_hidden_bias_arr_list = []
    
    def get_diff_hidden_bias_arr_list(self):
        ''' getter '''
        return self.__diff_hidden_bias_arr_list

    def set_diff_hidden_bias_arr_list(self, value):
        ''' setter '''
        self.__diff_hidden_bias_arr_list = value
    
    diff_hidden_bias_arr_list = property(get_diff_hidden_bias_arr_list, set_diff_hidden_bias_arr_list)

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

    def create_rnn_cells(
        self,
        int input_neuron_count,
        int hidden_neuron_count,
        int output_neuron_count
    ):
        self.hidden_activity_arr = np.zeros((1, hidden_neuron_count))
        self.rnn_activity_arr = np.zeros((1, hidden_neuron_count))

        self.weights_xg_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.01
        self.weights_xi_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.01
        self.weights_xf_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.01
        self.weights_xo_arr = np.random.normal(size=(input_neuron_count, hidden_neuron_count)) * 0.01

        self.weights_hg_arr = np.random.normal(size=(hidden_neuron_count, hidden_neuron_count)) * 0.01
        self.weights_hi_arr = np.random.normal(size=(hidden_neuron_count, hidden_neuron_count)) * 0.01
        self.weights_hf_arr = np.random.normal(size=(hidden_neuron_count, hidden_neuron_count)) * 0.01
        self.weights_ho_arr = np.random.normal(size=(hidden_neuron_count, hidden_neuron_count)) * 0.01

        self.weights_hy_arr = np.random.normal(size=(hidden_neuron_count, hidden_neuron_count)) * 0.01

        self.input_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.01
        self.hidden_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.01
        self.forget_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.01
        self.rnn_output_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.01
        self.linear_bias_arr = np.random.normal(size=hidden_neuron_count) * 0.01
        
        self.weights_hidden_output_arr = np.random.normal(size=(hidden_neuron_count, output_neuron_count)) * 0.01
        self.output_layer_bias_arr = np.random.normal(size=output_neuron_count) * 0.01

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
