# -*- coding: utf-8 -*-
import random
from pydbmmx.neuron_object import Neuron
from pydbmmx.neuron.interface.hidden_layer_interface import HiddenLayerInterface


class HiddenNeuron(Neuron, HiddenLayerInterface):
    '''
    The neurons in hidden layer.
    '''

    def hidden_update_state(self, link_value):
        '''
        Update activity.

        Args:
            link_value:      Input value.

        '''
        output = self.activate(link_value)
        self.activity = output

    def update_bias(self, learning_rate):
        '''
        Update biases.

        Args:
            learning_rate:  Learning rate.
        '''
        self.diff_bias += learning_rate * self.activity
