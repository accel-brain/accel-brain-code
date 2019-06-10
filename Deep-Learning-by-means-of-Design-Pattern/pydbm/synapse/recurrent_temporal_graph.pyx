# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse_list import Synapse
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.params_initializer import ParamsInitializer


class RecurrentTemporalGraph(Synapse):
    '''
    Recurrent Temporal Restricted Boltzmann Machines
    based on Complete Bipartite Graph.
    
    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.
    '''
    # Activity of neuron in visible layer.
    __visible_activity_arr = np.array([])

    def get_visible_activity_arr(self):
        ''' getter '''
        if isinstance(self.__visible_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __visible_activity_arr must be `np.ndarray`.")

        return self.__visible_activity_arr

    def set_visible_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __visible_activity_arr must be `np.ndarray`.")

        self.__visible_activity_arr = value

    visible_activity_arr = property(get_visible_activity_arr, set_visible_activity_arr)

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

    # Activity of neuron in hidden layer.
    __pre_hidden_activity_arr = np.array([])

    def get_pre_hidden_activity_arr(self):
        ''' getter '''
        if isinstance(self.__pre_hidden_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __pre_hidden_activity_arr must be `np.ndarray`.")
        return self.__pre_hidden_activity_arr

    def set_pre_hidden_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __pre_hidden_activity_arr must be `np.ndarray`.")
        self.__pre_hidden_activity_arr = value

    pre_hidden_activity_arr = property(get_pre_hidden_activity_arr, set_pre_hidden_activity_arr)

    # Bias of neuron in visible layer.
    __visible_bias_arr = np.array([])

    def get_visible_bias_arr(self):
        ''' getter '''
        if isinstance(self.__visible_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __visible_bias_arr must be `np.ndarray`.")

        return self.__visible_bias_arr

    def set_visible_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __visible_bias_arr must be `np.ndarray`.")

        self.__visible_bias_arr = value

    visible_bias_arr = property(get_visible_bias_arr, set_visible_bias_arr)

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

    # Diff of Bias of neuron in visible layer.
    __visible_diff_bias_arr = np.array([])

    def get_visible_diff_bias_arr(self):
        ''' getter '''
        if isinstance(self.__visible_diff_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __visible_diff_bias_arr must be `np.ndarray`.")

        return self.__visible_diff_bias_arr

    def set_visible_diff_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __visible_diff_bias_arr must be `np.ndarray`.")

        self.__visible_diff_bias_arr = value

    visible_diff_bias_arr = property(get_visible_diff_bias_arr, set_visible_diff_bias_arr)

    # Diff of Bias of neuron in hidden layer.
    __hidden_diff_bias_arr = np.array([])

    def get_hidden_diff_bias_arr(self):
        ''' getter '''
        if isinstance(self.__hidden_diff_bias_arr, np.ndarray) is False:
            raise TypeError("The type of __hidden_diff_bias_arr must be `np.ndarray`.")

        return self.__hidden_diff_bias_arr

    def set_hidden_diff_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __hidden_diff_bias_arr must be `np.ndarray`.")

        self.__hidden_diff_bias_arr = value

    hidden_diff_bias_arr = property(get_hidden_diff_bias_arr, set_hidden_diff_bias_arr)

    # Bias of neuron in hidden layer.
    __hat_hidden_activity_arr = np.array([])

    def get_hat_hidden_activity_arr(self):
        ''' getter '''
        if isinstance(self.__hat_hidden_activity_arr, np.ndarray) is False:
            raise TypeError("The type of __hat_hidden_activity_arr must be `np.ndarray`.")

        return self.__hat_hidden_activity_arr

    def set_hat_hidden_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __hat_hidden_activity_arr must be `np.ndarray`.")

        self.__hat_hidden_activity_arr = value

    hat_hidden_activity_arr = property(get_hat_hidden_activity_arr, set_hat_hidden_activity_arr)

    __inferenced_arr = np.array([])
    
    def get_inferenced_arr(self):
        ''' getter '''
        if isinstance(self.__inferenced_arr, np.ndarray) is False:
            raise TypeError("The type of __inferenced_arr must be `np.ndarray`.")
        return self.__inferenced_arr

    def set_inferenced_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of value must be `np.ndarray`.")

        self.__inferenced_arr = value
    
    inferenced_arr = property(get_inferenced_arr, set_inferenced_arr)

    # Activation function in visible layer.
    def get_visible_activating_function(self):
        ''' getter '''
        if isinstance(self.shallower_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __visible_activating_function must be `ActivatingFunctionInterface`.")
        return self.shallower_activating_function

    def set_visible_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __visible_activating_function must be `ActivatingFunctionInterface`.")
        self.shallower_activating_function = value

    visible_activating_function = property(get_visible_activating_function, set_visible_activating_function)

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

    # Activation function in hidden layer.
    __rnn_activating_function = None

    def get_rnn_activating_function(self):
        ''' getter '''
        if isinstance(self.__rnn_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __hidden_activating_function must be `ActivatingFunctionInterface`.")
        return self.__rnn_activating_function

    def set_rnn_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __hidden_activating_function must be `ActivatingFunctionInterface`.")
        self.__rnn_activating_function = value

    rnn_activating_function = property(get_rnn_activating_function, set_rnn_activating_function)

    # Weights matrix in visible layer for RNN.
    __rnn_visible_weights_arr = np.array([])

    def get_rnn_visible_weights_arr(self):
        ''' getter '''
        if isinstance(self.__rnn_visible_weights_arr, np.ndarray) is False:
            raise TypeError()
        return self.__rnn_visible_weights_arr

    def set_rnn_visible_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__rnn_visible_weights_arr = value

    rnn_visible_weights_arr = property(get_rnn_visible_weights_arr, set_rnn_visible_weights_arr)

    # Weights matrix in hidden layer for RNN.
    __rnn_hidden_weights_arr = np.array([])

    def get_rnn_hidden_weights_arr(self):
        ''' getter '''
        if isinstance(self.__rnn_hidden_weights_arr, np.ndarray) is False:
            raise TypeError()
        return self.__rnn_hidden_weights_arr

    def set_rnn_hidden_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__rnn_hidden_weights_arr = value

    rnn_hidden_weights_arr = property(get_rnn_hidden_weights_arr, set_rnn_hidden_weights_arr)

    # bias in visible layer for RNN.
    __rnn_visbile_bias_arr = np.array([])

    def get_rnn_visbile_bias_arr(self):
        ''' getter '''
        if isinstance(self.__rnn_visbile_bias_arr, np.ndarray) is False:
            raise TypeError()
        return self.__rnn_visbile_bias_arr

    def set_rnn_visible_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__rnn_visbile_bias_arr = value

    rnn_visible_bias_arr = property(get_rnn_visbile_bias_arr, set_rnn_visible_bias_arr)

    # bias in hidden layer for RNN.
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
            weights_arr:                        The pre-learned weights of links.
                                                If this array is not empty, `ParamsInitializer.sample_f` will not be called 
                                                and `weights_arr` will be refered as initial weights.

            scale:                              Scale of parameters which will be `ParamsInitializer`.
            params_initializer:                 is-a `ParamsInitializer`.
            params_dict:                        `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.
        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        self.visible_bias_arr = np.zeros((shallower_neuron_count, ))
        self.hidden_bias_arr = np.zeros((deeper_neuron_count, ))
        self.visible_diff_bias_arr = np.zeros(self.visible_bias_arr.shape)
        self.hidden_diff_bias_arr = np.zeros(self.hidden_bias_arr.shape)

        self.rnn_visible_weights_arr = params_initializer.sample(
            size=(shallower_neuron_count, deeper_neuron_count),
            **params_dict
        ) * scale
        self.rnn_hidden_weights_arr = params_initializer.sample(
            size=(deeper_neuron_count, deeper_neuron_count),
            **params_dict
        ) * scale

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
