# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse.recurrent_temporal_graph import RecurrentTemporalGraph
from pydbm.params_initializer import ParamsInitializer


class RNNGraph(RecurrentTemporalGraph):
    '''
    Recurrent Neural Network Restricted Boltzmann Machines (RNN-RBM)
    based on Complete Bipartite Graph.
    
    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.
    '''

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
        if isinstance(self.__diff_visible_bias_arr_list, list) is False:
            raise TypeError()
        return self.__diff_visible_bias_arr_list

    def set_diff_visible_bias_arr_list(self, value):
        ''' setter '''
        if isinstance(value, list) is False:
            raise TypeError()
        self.__diff_visible_bias_arr_list = value
    
    diff_visible_bias_arr_list = property(get_diff_visible_bias_arr_list, set_diff_visible_bias_arr_list)

    __diff_hidden_bias_arr_list = []
    
    def get_diff_hidden_bias_arr_list(self):
        ''' getter '''
        if isinstance(self.__diff_hidden_bias_arr_list, list) is False:
            raise TypeError()
        return self.__diff_hidden_bias_arr_list

    def set_diff_hidden_bias_arr_list(self, value):
        ''' setter '''
        if isinstance(value, list) is False:
            raise TypeError()
        self.__diff_hidden_bias_arr_list = value
    
    diff_hidden_bias_arr_list = property(get_diff_hidden_bias_arr_list, set_diff_hidden_bias_arr_list)

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
        self.rnn_hidden_bias_arr = np.zeros((deeper_neuron_count, ))

        self.rnn_visible_weights_arr = params_initializer.sample(
            size=(shallower_neuron_count, deeper_neuron_count),
            **params_dict
        ) * scale
        self.rnn_hidden_weights_arr = params_initializer.sample(
            size=(deeper_neuron_count, deeper_neuron_count),
            **params_dict
        ) * scale

        self.diff_rnn_hidden_weights_arr = np.zeros(
            (deeper_neuron_count, deeper_neuron_count)
        )

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
