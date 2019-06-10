# -*- coding: utf-8 -*-
import os
import numpy as np
cimport numpy as np
cimport cython
import random
ctypedef np.float64_t DOUBLE_t
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.optimization.batch_norm import BatchNorm
from pydbm.params_initializer import ParamsInitializer


class Synapse(object):
    '''
    The object of synapse.
    '''
    
    # The weights of links
    __weights_arr = np.array([])

    def get_weights_arr(self):
        ''' getter '''
        if isinstance(self.__weights_arr, np.ndarray) is False:
            raise TypeError("The type of __weights_arr must be `np.ndarray`.")
        return self.__weights_arr

    def set_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __weights_arr must be `np.ndarray`.")
        self.__weights_arr = value

    weights_arr = property(get_weights_arr, set_weights_arr)

    # The diff of weights.
    __diff_weights_arr = np.array([])

    def get_diff_weights_arr(self):
        ''' getter '''
        if isinstance(self.__diff_weights_arr, np.ndarray) is False:
            raise TypeError("The type of __diff_weights_arr must be `np.ndarray`.")
        return self.__diff_weights_arr

    def set_diff_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError("The type of __diff_weights_arr must be `np.ndarray`.")
        self.__diff_weights_arr = value

    diff_weights_arr = property(get_diff_weights_arr, set_diff_weights_arr)

    # Activation function in shallower layer.
    __shallower_activating_function = None

    def get_shallower_activating_function(self):
        ''' getter '''
        if isinstance(self.__shallower_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __shallower_activating_function must be `ActivatingFunctionInterface`.")
        return self.__shallower_activating_function

    def set_shallower_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __shallower_activating_function must be `ActivatingFunctionInterface`.")

        self.__shallower_activating_function = value

    shallower_activating_function = property(get_shallower_activating_function, set_shallower_activating_function)

    # Activation function in deeper layer.
    __deeper_activating_function = None

    def get_deeper_activating_function(self):
        ''' getter '''
        if isinstance(self.__deeper_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __deeper_activating_function must be `ActivatingFunctionInterface`.")
        return self.__deeper_activating_function

    def set_deeper_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of __deeper_activating_function must be `ActivatingFunctionInterface`.")
        self.__deeper_activating_function = value

    deeper_activating_function = property(get_deeper_activating_function, set_deeper_activating_function)

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

        self.shallower_activating_function = shallower_activating_function
        self.deeper_activating_function = deeper_activating_function

        if weights_arr.shape[0]:
            self.weights_arr = weights_arr
        else:
            self.weights_arr = params_initializer.sample(
                size=(shallower_neuron_count, deeper_neuron_count),
                **params_dict
            ) * scale
        self.diff_weights_arr = np.zeros(self.weights_arr.shape, dtype=float)
        self.stacked_graph_list = []

    def learn_weights(self):
        '''
        Update the weights of links.
        '''
        self.weights_arr = self.weights_arr + self.diff_weights_arr
        cdef int row = self.weights_arr.shape[0]
        cdef int col = self.weights_arr.shape[1]
        self.diff_weights_arr = np.zeros((row, col), dtype=float)

    def save_pre_learned_params(self, file_path):
        '''
        Save pre-learned parameters.

        If you want to save pre-learned parameters simultaneously with stacked graphs,
        call method `stack_graph` and setup the graphs before calling this method.

        If this class's subclass has a `ActivatingFunctionInterface` which has a `BatchNorm`, 
        this class store `BatchNorm`'s `beta_arr` and `gamma_arr` to your file.

        Args:
            file_path:    File path.
        '''
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d.setdefault(k, v)
            if isinstance(v, ActivatingFunctionInterface) is True:
                if v.batch_norm is not None and isinstance(v.batch_norm, BatchNorm) is True:
                    d.setdefault(k + "_batch_norm_beta", v.batch_norm.beta_arr)
                    d.setdefault(k + "_batch_norm_gamma", v.batch_norm.gamma_arr)

        np.savez_compressed(file_path, **d)

    def load_pre_learned_params(self, file_path):
        '''
        Load pre-learned parameters.

        If you want to load pre-learned parameters simultaneously with stacked graphs,
        call method `stack_graph` and setup the graphs before calling this method.

        If this class's subclass has a `ActivatingFunctionInterface` which has a `BatchNorm`, 
        and your file stores appropriate data, this class set `BatchNorm`'s `beta_arr` and `gamma_arr`.

        Args:
            file_path:    File path.
        '''
        pre_learned_dict = np.load(file_path)
        for k, v in pre_learned_dict.items():
            if isinstance(v, np.ndarray):
                self.__dict__[k] = v

        for k, v in self.__dict__.items():
            if isinstance(v, ActivatingFunctionInterface) is True:
                if v.batch_norm is not None and isinstance(v.batch_norm, BatchNorm) is True:
                    if k + "_batch_norm_beta" in pre_learned_dict:
                        self.__dict__[k].batch_norm.beta_arr = pre_learned_dict[k + "_batch_norm_beta"]
                    if k + "_batch_norm_gamma" in pre_learned_dict:
                        self.__dict__[k].batch_norm.gamma_arr = pre_learned_dict[k + "_batch_norm_gamma"]
