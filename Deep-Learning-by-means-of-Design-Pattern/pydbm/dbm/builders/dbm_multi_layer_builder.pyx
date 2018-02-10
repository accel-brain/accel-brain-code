# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.synapse.complete_bipartite_graph import CompleteBipartiteGraph
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine


class DBMMultiLayerBuilder(DBMBuilder):
    '''
    `Concrete Builder` in Builder Pattern.

    Compose three restricted boltzmann machines for building a deep boltzmann machine.
    '''
    # The list of neurons in visible layer.
    __visible_neuron_list = []
    # The list of neurons for feature points in `virtual` visible layer. 
    __feature_point_neuron = []
    # the list of neurons in hidden layer.
    __hidden_neuron_list = []
    # Complete bipartite graph
    __graph_list = []
    # The list of restricted boltzmann machines.
    __rbm_list = []
    # Learning rate.
    __learning_rate = 0.5

    def get_learning_rate(self):
        ''' getter '''
        if isinstance(self.__learning_rate, float) is False:
            raise TypeError()
        return self.__learning_rate

    def set_learning_rate(self, value):
        ''' setter '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__learning_rate = value

    learning_rate = property(get_learning_rate, set_learning_rate)

    # Dropout rate.
    __dropout_rate = 0.5

    def get_dropout_rate(self):
        ''' getter '''
        if isinstance(self.__dropout_rate, float) is False:
            raise TypeError()
        return self.__dropout_rate

    def set_dropout_rate(self, value):
        ''' setter '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__dropout_rate = value

    dropout_rate = property(get_dropout_rate, set_dropout_rate)

    def __init__(self):
        '''
        Initialize.
        '''
        self.__visible_neuron_list = []
        self.__feature_point_neuron = []
        self.__hidden_neuron_list = []
        self.__graph_list = []
        self.__rbm_list = []

    def visible_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in visible layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        self.__visible_activating_function = activating_function
        self.__visible_neuron_count = neuron_count

    def feature_neuron_part(self, activating_function_list, neuron_count_list):
        '''
        Build neurons for feature points in `virtual` visible layer.

        Build neurons in `n` layers.

        For associating with `n-1` layers, the object activate as neurons in hidden layer.
        On the other hand, for associating with `n+1` layers, the object activate as neurons in `virtual` visible layer.

        Args:
            activating_function_list:    The list of activation function.
            neuron_count_list:           The list of the number of neurons.
        '''
        self.__feature_activating_function_list = activating_function_list
        self.__feature_point_count_list = neuron_count_list

    def hidden_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activating_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        self.__hidden_activating_function = activating_function
        self.__hidden_neuron_list = neuron_count

    def graph_part(self, approximate_interface):
        '''
        Build complete bipartite graph.

        Args:
            approximate_interface:       The object of function approximation.
        '''
        complete_bipartite_graph = CompleteBipartiteGraph()
        complete_bipartite_graph.create_node(
            self.__visible_neuron_count,
            self.__feature_point_count_list[0],
            self.__visible_activating_function,
            self.__feature_activating_function_list[0]
        )
        self.__graph_list.append(complete_bipartite_graph)

        cdef int i
        for i in range(1, len(self.__feature_point_count_list)):
            complete_bipartite_graph = CompleteBipartiteGraph()
            complete_bipartite_graph.create_node(
                self.__feature_point_count_list[i - 1],
                self.__feature_point_count_list[i],
                self.__feature_activating_function_list[i - 1],
                self.__feature_activating_function_list[i]
            )
            self.__graph_list.append(complete_bipartite_graph)

        complete_bipartite_graph = CompleteBipartiteGraph()
        complete_bipartite_graph.create_node(
            self.__feature_point_count_list[-1],
            self.__hidden_neuron_list,
            self.__feature_activating_function_list[-1],
            self.__hidden_activating_function
        )
        self.__graph_list.append(complete_bipartite_graph)

    def get_result(self):
        '''
        Return builded restricted boltzmann machines.

        Returns:
            The list of restricted boltzmann machines.

        '''
        for graph in self.__graph_list:
            rbm = RestrictedBoltzmannMachine(
                graph,
                self.__learning_rate,
                self.__dropout_rate,
                ContrastiveDivergence()
            )
            self.__rbm_list.append(rbm)

        return self.__rbm_list
