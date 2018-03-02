# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class RTRBMBuilder(metaclass=ABCMeta):
    '''
    The `Builder` interface in Builder Pattern, which generates the object of RTRBM.
    '''

    @abstractproperty
    def learning_rate(self):
        ''' Learning rate. '''
        raise NotImplementedError()

    @abstractmethod
    def visible_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in visible layer.

        Args:
            activation_function:    Activating function.
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError()

    @abstractmethod
    def hidden_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in hidden layers.

        Args:
            activation_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError

    @abstractmethod
    def graph_part(self, approximate_interface):
        '''
        Build RTRBM graph.

        Args:
            approximate_interface:    The function approximation.
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_result():
        '''
        Return builded restricted boltzmann machines.

        Returns:
            The list of restricted boltzmann machines.
        '''
        raise NotImplementedError()
