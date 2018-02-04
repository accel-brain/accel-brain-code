# -*- coding: utf-8 -*-
import numpy as np
import random
import mxnet as mx


class Synapse(object):
    '''
    The object of synapse.
    '''
    # The list of nuron's object in shallowr layer.
    __shallower_activity_arr = None
    # The list of neuron's object in deeper layer.
    __deeper_activity_arr = None
    # `nd.array` of the weights.
    __weights_arr = None
    # `nd.array` of the difference of weights.
    __diff_weights_arr = None

    def get_shallower_activity_arr(self):
        ''' getter '''
        return self.__shallower_activity_arr

    def set_shallower_activity_arr(self, value):
        ''' setter '''
        self.__shallower_activity_arr = value

    def get_deeper_activity_arr(self):
        ''' getter '''
        return self.__deeper_activity_arr

    def set_deeper_activity_arr(self, value):
        ''' setter '''
        self.__deeper_activity_arr = value

    def get_weights_arr(self):
        ''' getter '''
        return self.__weights_arr

    def set_weights_arr(self, value):
        ''' setter '''
        self.__weights_arr = value

    def get_diff_weights_arr(self):
        ''' getter '''
        return self.__diff_weights_arr

    def set_diff_weights_arr(self, value):
        ''' setter '''
        self.__diff_weights_arr = value

    shallower_activity_arr = property(get_shallower_activity_arr, set_shallower_activity_arr)
    deeper_activity_arr = property(get_deeper_activity_arr, set_deeper_activity_arr)
    weights_arr = property(get_weights_arr, set_weights_arr)
    diff_weights_arr = property(get_diff_weights_arr, set_diff_weights_arr)

    def create_node(
        self,
        shallower_activity_arr,
        deeper_activity_arr,
        weights_arr=None
    ):
        '''
        Set links of nodes to the graphs.

        Args:
            shallower_activity_arr:      The list of neuron's object in shallowr layer.
            deeper_activity_arr:         The list of neuron's object in deeper layer.
            weights_arr:                `nd.array` of the weights.
        '''
        self.__shallower_activity_arr = shallower_activity_arr
        self.__deeper_activity_arr = deeper_activity_arr

        init_weights_arr = mx.ndarray.random.uniform(
            shape=(
                self.__shallower_activity_arr.shape[0],
                self.__deeper_activity_arr.shape[0]
            )
        )
        if weights_arr is not None:
            self.weights_arr = weights_arr
        else:
            self.weights_arr = init_weights_arr

    def learn_weights(self):
        '''
        Update the weights of links.
        '''
        self.weights_arr = self.weights_arr + self.diff_weights_arr
        row = self.weights_arr.shape[0]
        col = self.weights_arr.shape[1]
        self.diff_weights_arr = mx.nd.zeros((row, col), dtype=float)

    def normalize_visible_bias(self):
        '''
        Normalize the neuron's activity in visible layers.
        '''
        visible_activity_arr = self.__shallower_activity_arr
        if visible_activity_arr.shape[0] > 1 and mx.nd.sum(visible_activity_arr) != 0:
            visible_activity_arr = visible_activity_arr / mx.nd.sum(visible_activity_arr)
        self.__shallower_activity_arr = visible_activity_arr

    def normalize_hidden_bias(self):
        '''
        normalize the neuron's activity in hidden layers.
        '''
        hidden_activity_arr = self.__deeper_activity_arr
        if hidden_activity_arr.shape[0] > 1 and mx.nd.sum(hidden_activity_arr) != 0:
            hidden_activity_arr = hidden_activity_arr / mx.nd.sum(hidden_activity_arr)
        self.__deeper_activity_arr = hidden_activity_arr
