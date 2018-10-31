# -*- coding: utf-8 -*-
from pydbm.dbm.interface.rt_rbm_builder import RTRBMBuilder
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class RTRBMDirector(object):
    '''
    The `Director` in Builder Pattern.

    Compose RTRBM, RNN-RBM, or LSTM-RTRBM for building a object of restricted boltzmann machine.
    '''

    # `Builder` in Builder Pattern.
    __rtrbm_builder = None
    # The restricted boltzmann machines.
    __rbm = None
    
    def get_rbm(self):
        ''' getter '''
        return self.__rbm

    def set_rbm(self, value):
        ''' setter '''
        self.__rbm = value
    
    rbm = property(get_rbm, set_rbm)
    
    def __init__(self, rtrbm_builder):
        '''
        Initialize `Builder` in Builder Pattern.

        Args:
            rtrbm_builder     `Concreat Builder` in Builder Pattern
        '''
        if isinstance(rtrbm_builder, RTRBMBuilder):
            self.__rtrbm_builder = rtrbm_builder
        else:
            raise TypeError()

        self.__rbm = None

    def rtrbm_construct(
        self,
        visible_num,
        hidden_num,
        visible_activating_function,
        hidden_activating_function,
        rnn_activating_function,
        approximate_interface,
        learning_rate=1e-05
    ):
        '''
        Build deep boltzmann machine.

        Args:
            visible_num:                    The number of units in visible layer.
            hidden_num:                     The number of units in hidden layer.
            visible_activating_function:    The activation function in visible layer.
            hidden_activating_function:     The activation function in hidden layer.
            approximate_interface:          The function approximation.
        '''
        if isinstance(visible_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        if isinstance(hidden_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        if isinstance(rnn_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        if isinstance(approximate_interface, ApproximateInterface) is False:
            raise TypeError()

        # Learning rate.
        self.__rtrbm_builder.learning_rate = learning_rate
        # Set units in visible layer.
        self.__rtrbm_builder.visible_neuron_part(
            visible_activating_function,
            visible_num
        )
        # Set units in hidden layer.
        self.__rtrbm_builder.hidden_neuron_part(
            hidden_activating_function,
            hidden_num
        )
        # Set units in RNN layer.
        self.__rtrbm_builder.rnn_neuron_part(rnn_activating_function)
        # Set graph and approximation function, delegating `SGD` which is-a `OptParams`.
        self.__rtrbm_builder.graph_part(approximate_interface)
        # Building.
        self.__rbm = self.__rtrbm_builder.get_result()
