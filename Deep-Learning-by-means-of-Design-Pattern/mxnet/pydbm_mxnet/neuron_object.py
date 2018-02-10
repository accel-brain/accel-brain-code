# -*- coding: utf-8 -*-
import mxnet as mx
from abc import ABCMeta, abstractmethod, abstractproperty
from pydbm_mxnet.synapse_list import Synapse
from pydbm_mxnet.activation.interface.activating_function_interface import ActivatingFunctionInterface


class Neuron(metaclass=ABCMeta):
    '''
    Template Method Pattern of neuron.
    '''
    # Node index.
    __node_index = None
    # Bias.
    __bias = 0.0
    # The ndarray of bias.
    __bias_arr = None
    # The difference of bias.
    __diff_bias = 0.0
    # The ndarray of difference of bias.
    __diff_bias_arr = None
    # Activation function.
    __activating_function = None
    # Synapse.
    __synapse_list = None

    def get_node_index(self):
        ''' getter '''
        if isinstance(self.__node_index, int) is False:
            raise TypeError()
        return self.__node_index

    def set_node_index(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError()
        self.__node_index = value

    def get_activating_function(self):
        ''' getter of activating_function '''
        if isinstance(self.__activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.__activating_function

    def set_activating_function(self, value):
        ''' setter of activating_function '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.__activating_function = value

    def get_synapse_list(self):
        ''' getter '''
        if isinstance(self.__synapse_list, Synapse) is False:
            raise TypeError()
        return self.__synapse_list

    def set_synapse_list(self, value):
        ''' setter '''
        if isinstance(value, Synapse) is False:
            raise TypeError()
        self.__synapse_list = value

    node_index = property(get_node_index, set_node_index)

    @abstractproperty
    def activity_arr(self):
        ''' Activity array.'''
        raise NotImplementedError()

    @abstractproperty
    def bias(self):
        ''' bias. '''
        raise NotImplementedError()

    @abstractproperty
    def bias_arr(self):
        ''' bias array. '''
        raise NotImplementedError()

    activating_function = property(get_activating_function, set_activating_function)

    @abstractproperty
    def activity(self):
        ''' Activity. '''
        raise NotImplementedError()

    synapse_list = property(get_synapse_list, set_synapse_list)

    # Alias of `synapse_list`.
    graph = property(get_synapse_list, set_synapse_list)

    def activate(self, link_value):
        '''
        Activate.

        Args:
            link_value    input value to activation function.

        Returns:
            true: activation. false: not activation.
        '''
        output = self.activating_function.activate(
            link_value + self.bias
        )
        return output

    @abstractmethod
    def update_bias(self, learning_rate):
        '''
        Update bias with the difference.

        Args:
            learning_rate:  Learning rate.
        '''
        raise NotImplementedError()

    def learn_bias(self):
        '''
        Learn with bias.
        '''
        self.bias += self.diff_bias
        self.diff_bias = 0.0
