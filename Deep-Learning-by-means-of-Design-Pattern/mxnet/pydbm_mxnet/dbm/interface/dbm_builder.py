# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class DBMBuilder(metaclass=ABCMeta):
    '''
    The `Builder` interface in Builder Pattern, which generates the object of DBM.
    '''

    @abstractproperty
    def learning_rate(self):
        ''' Learning rate '''
        raise NotImplementedError()

    @abstractmethod
    def visible_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in visible layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError()

    @abstractmethod
    def feature_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons for feature points in `virtual` visible layer.

        Build neurons in `n` layers.

        For associating with `n-1` layers, the object activate as neurons in hidden layer.
        On the other hand, for associating with `n+1` layers, the object activate as neurons in `virtual` visible layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError()

    @abstractmethod
    def hidden_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activating_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        raise NotImplementedError

    @abstractmethod
    def graph_part(self, approximate_interface):
        '''
        Build complete bipartite graph.

        Args:
            approximate_interface:       The object of function approximation.
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
