# -*- coding: utf-8 -*-
import mxnet as mx
from abc import ABCMeta, abstractmethod
from pydbmmx.activation.interface.activating_function_interface import ActivatingFunctionInterface


class Neuron(metaclass=ABCMeta):
    '''
    Template Method Pattern of neuron.
    '''
    # Node index.
    __node_index = None
    # The ndarray of activity.
    __activity_arr = None
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

    def get_activity(self):
        ''' getter of activity'''
        if isinstance(self.activity_arr[self.node_index], float) is False:
            raise TypeError()
        return self.activity_arr[self.node_index]

    def set_activity(self, value):
        ''' setter of activity '''
        if isinstance(value, float) is False and isinstance(value, int) is False:
            raise TypeError()
        self.activity_arr[self.node_index] = float(value)

    def get_activating_arr(self):
        ''' getter '''
        if isinstance(self.__activity_arr, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()
        return self.__activity_arr

    def set_activity_arr(self, value):
        ''' setter '''
        if isinstance(value, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()
        self.__activity_arr = value

    def get_bias(self):
        ''' getter of bias '''
        if isinstance(self.bias_arr[self.node_index], float) is False:
            raise TypeError()
        return self.bias_arr[self.node_index]

    def set_bias(self, value):
        ''' setter of bias '''
        if isinstance(self.bias_arr[self.node_index], float) is False:
            raise TypeError()
        self.bias_arr[self.node_index] = value

    def get_bias_arr(self):
        ''' getter '''
        if isinstance(self.__bias_arr, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()
        return self.__bias_arr

    def set_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()
        self.__bias_arr = value

    def get_diff_bias(self):
        ''' getter of diff_bias '''
        if isinstance(self.diff_bias_arr[self.node_index], float) is False:
            raise TypeError()
        return self.diff_bias_arr[self.node_index]

    def set_diff_bias(self, value):
        ''' setter of diff_bias '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.diff_bias_arr[self.node_index] = value

    def get_diff_bias_arr(self):
        ''' getter '''
        if isinstance(self.__diff_bias_arr, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()
        return self.__diff_bias_arr

    def set_diff_bias_arr(self, value):
        ''' setter '''
        if isinstance(value, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()
        self.__diff_bias_arr = value

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

    node_index = property(get_node_index, set_node_index)
    activity_arr = property(get_activating_arr, set_activity_arr)
    bias = property(get_bias, set_bias)
    bias_arr = property(get_bias_arr, set_bias_arr)
    diff_bias = property(get_diff_bias, set_diff_bias)
    diff_bias_arr = property(get_diff_bias_arr, set_diff_bias_arr)
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
