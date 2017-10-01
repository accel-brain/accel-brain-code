# -*- coding: utf-8 -*-
import pyximport; pyximport.install()
from abc import ABCMeta, abstractmethod


class NNBuilder(metaclass=ABCMeta):
    '''
    `Builder` in Builder Pattern.
    
    Compose graphs of synapse for building the object of neural networks.
    '''

    @abstractmethod
    def input_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in input layer.

        Args:
            activation_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError()

    @abstractmethod
    def hidden_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activation_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError()

    @abstractmethod
    def output_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in output layer.

        Args:
            activation_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError

    @abstractmethod
    def graph_part(self, approximate_interface):
        '''
        Build graphs of synapse.

        Args:
            approximate_interface:    The object of function approximation.
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_result():
        '''
        Return the list of builded graphs of synapse.

        Returns:
            The list of graphs of synapse.
        '''
        raise NotImplementedError()
