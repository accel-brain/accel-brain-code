# -*- coding: utf-8 -*-
import numpy as np
from abc import ABCMeta, abstractmethod
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class Neuron(metaclass=ABCMeta):
    '''
    Template Method Pattern of neuron.
    '''
    # Node index.
    __node_index = None
    # Activity List.
    __activity_list = []
    # Bias.
    __bias = 0.0
    # The difference of bias.
    __diff_bias = 0.0
    # Activation function.
    __activating_function = None

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

    def get_activating_list(self):
        ''' getter '''
        if isinstance(self.__activity_list, list) is False:
            raise TypeError()
        return self.__activity_list

    def set_activity_list(self, value):
        ''' setter '''
        if isinstance(value, list) is False:
            raise TypeError()
        self.__activity_list = value

    def get_bias(self):
        ''' getter of bias '''
        if isinstance(self.__bias, float) is False:
            raise TypeError()
        return self.__bias

    def set_bias(self, double value):
        ''' setter of bias '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__bias = value

    def get_diff_bias(self):
        ''' getter of diff_bias '''
        if isinstance(self.__diff_bias, float) is False:
            raise TypeError()
        return self.__diff_bias

    def set_diff_bias(self, double value):
        ''' setter of diff_bias '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__diff_bias = value

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

    def get_activity(self):
        ''' getter of activity'''
        if isinstance(self.activity_list[self.node_index], float) is False:
            raise TypeError()
        return self.activity_list[self.node_index]

    def set_activity(self, value):
        ''' setter of activity '''
        if isinstance(value, float) is False and isinstance(value, int) is False:
            raise TypeError()
        self.activity_list[self.node_index] = float(value)

    node_index = property(get_node_index, set_node_index)
    activity_list = property(get_activating_list, set_activity_list)
    bias = property(get_bias, set_bias)
    diff_bias = property(get_diff_bias, set_diff_bias)
    activating_function = property(get_activating_function, set_activating_function)
    activity = property(get_activity, set_activity)

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
    def update_bias(self, double learning_rate):
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
