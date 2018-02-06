# -*- coding: utf-8 -*-
from pydbmmx.dbm.interface.dbm_builder import DBMBuilder
from pydbmmx.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
from pydbmmx.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbmmx.approximation.interface.approximate_interface import ApproximateInterface


class DBMDirector(object):
    '''
    The `Director` in Builder Pattern.
    
    Compose restricted boltzmann machines for building a object of deep boltzmann machine.
    '''

    # `Builder` in Builder Pattern.
    __dbm_builder = None
    # The list of restricted boltzmann machines.
    __rbm_list = []

    def get_rbm_list(self):
        ''' getter '''
        if isinstance(self.__rbm_list, list) is False:
            raise TypeError()

        for rbm in self.__rbm_list:
            if isinstance(rbm, RestrictedBoltzmannMachine) is False:
                raise TypeError()

        return self.__rbm_list

    def set_rbm_list(self, value):
        ''' setter '''
        if isinstance(value, list) is False:
            raise TypeError()

        for rbm in value:
            if isinstance(rbm, RestrictedBoltzmannMachine) is False:
                raise TypeError()

        self.__rbm_list = value

    rbm_list = property(get_rbm_list, set_rbm_list)

    def __init__(self, dbm_builder):
        '''
        Initialize `Builder` in Builder Pattern.

        Args:
            dbm_builder     `Concreat Builder` in Builder Pattern
        '''
        if isinstance(dbm_builder, DBMBuilder) is False:
            raise TypeError()

        self.__dbm_builder = dbm_builder

    def dbm_construct(
        self,
        neuron_assign_list,
        activating_function,
        approximate_interface
    ):
        '''
        Build deep boltzmann machine.

        Args:
            neuron_assign_list:     The unit of neurons in each layers.
            activating_function:    Activation function,
            approximate_interface:  The object of function approximation.
        '''
        if isinstance(activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()

        if isinstance(approximate_interface, ApproximateInterface) is False:
            raise TypeError()

        visible_neuron_count = neuron_assign_list[0]
        hidden_neuron_count = neuron_assign_list[-1]

        self.__dbm_builder.visible_neuron_part(activating_function, visible_neuron_count)

        for i in range(1, len(neuron_assign_list) - 1):
            feature_neuron_count = neuron_assign_list[i]
            self.__dbm_builder.feature_neuron_part(activating_function, feature_neuron_count)

        self.__dbm_builder.hidden_neuron_part(activating_function, hidden_neuron_count)
        self.__dbm_builder.graph_part(approximate_interface)
        self.rbm_list = self.__dbm_builder.get_result()
