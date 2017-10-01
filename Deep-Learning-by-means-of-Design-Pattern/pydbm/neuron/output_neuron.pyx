# -*- coding: utf-8 -*-
import pyximport;pyximport.install()
import random
import bisect
from pydbm.neuron_object import Neuron
from pydbm.neuron.interface.output_layer_interface import OutputLayerInterface


class OutputNeuron(Neuron, OutputLayerInterface):
    '''
    The neurons in output layer.
    '''

    # If True, `Self` set the activity as binaly.
    __bernoulli_flag = False

    def get_bernoulli_flag(self):
        ''' getter '''
        if isinstance(self.__bernoulli_flag, bool) is False:
            raise TypeError()
        return self.__bernoulli_flag

    def set_bernoulli_flag(self, value):
        ''' setter '''
        if isinstance(value, bool) is False:
            raise TypeError()
        self.__bernoulli_flag = value

    bernoulli_flag = property(get_bernoulli_flag, set_bernoulli_flag)

    def __init__(self):
        '''
        Initialize.
        '''
        self.bias = round(random.random(), 3)

    def output_update_state(self, double link_value):
        '''
        Update activity.

        Args:
            link_value:      Input value.

        '''
        output = self.activate(link_value)
        if self.bernoulli_flag is False:
            self.activity = output
        else:
            activated_flag = self.__decide_activation(output)
            if activated_flag:
                self.activity = 1.0
            else:
                self.activity = 0.0

    def update_bias(self, double learning_rate):
        '''
        Update biases.

        Args:
            learning_rate:  Learning rate.
        '''
        self.diff_bias += learning_rate * self.activity

    def __decide_activation(self, double probabirity):
        '''
        Decide the binaly activity.

        Args:
            probabirity:    Activity.

        Returns:
            If True, it is activated.
        '''
        probabirity_list, result_list = [probabirity, 1.0], [True, False]
        return result_list[bisect.bisect(probabirity_list, random.random())]

    def release(self):
        '''
        Release the activity.

        Returns:
            Activity.
        '''
        return self.activity
