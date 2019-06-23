# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty
from pydbm.params_initializer import ParamsInitializer


class DBMBuilder(metaclass=ABCMeta):
    '''
    The `Builder` interface in Builder Pattern, which generates the object of DBM.
    '''

    @abstractproperty
    def learning_rate(self):
        ''' Learning rate. '''
        raise NotImplementedError()

    @abstractproperty
    def learning_attenuate_rate(self):
        '''
        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
        '''
        raise NotImplementedError()

    @abstractproperty
    def attenuate_epoch(self):
        '''
        Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
        '''
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
    def feature_neuron_part(self, activating_function_list, neuron_count_list):
        '''
        Build neurons for feature points in `virtual` visbile layer.

        Build neurons in `n` layers.

        For associating with `n-1` layers, the object activates as neurons in hidden layer.
        On the other hand, for associating with `n+1` layers, the object activates as neurons in `virtual` visible layer.

        Args:
            activation_function_list:    The list of activation function.
            neuron_count_list:           The list of the number of neurons.
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
    def graph_part(
        self, 
        approximate_interface_list,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Build complete bipartite graph.

        Args:
            approximate_interface_list:     The list of function approximation.
            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

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
