# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
cimport numpy
from abc import ABCMeta, abstractmethod
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class Neuron(metaclass=ABCMeta):
    '''
    Template Method Pattern of neuron.
    '''

    # Bias.
    __bias = 0.0
    # The difference of bias.
    __diff_bias = 0.0
    # Activation function.
    __activating_function = None
    # The activity.
    __activity = 0.0

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
        if isinstance(self.__activity, float) is False:
            raise TypeError()
        return self.__activity

    def set_activity(self, double value):
        ''' setter of activity '''
        if isinstance(value, float) is False and isinstance(value, int) is False:
            raise TypeError()
        self.__activity = float(value)

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
        cdef double output = self.activating_function.activate(
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
