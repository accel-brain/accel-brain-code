# -*- coding: utf-8 -*-

import numpy as np

cimport numpy as np
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.neuron.visible_neuron import VisibleNeuron
from pydbm.neuron.hidden_neuron import HiddenNeuron
from pydbm.neuron.feature_point_neuron import FeaturePointNeuron
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.synapse.complete_bipartite_graph import CompleteBipartiteGraph
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine


class DBMMultiLayerBuilder(DBMBuilder):
    '''
    `Concrete Builder` in Builder Pattern.

    Compose three restricted boltzmann machines for building a deep boltzmann machine.
    '''
    # The list of neurons in visible layer.
    __visual_neuron_list = []
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

    def __init__(self):
        '''
        Initialize.
        '''
        self.__visual_neuron_list = []
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
        cdef int i
        for i in range(neuron_count):
            visible_neuron = VisibleNeuron()
            visible_neuron.activating_function = activating_function
            visible_neuron.bernoulli_flag = True
            self.__visual_neuron_list.append(visible_neuron)

    def feature_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons for feature points in `virtual` visible layer.

        Build neurons in `n` layers.

        For associating with `n-1` layers, the object activate as neurons in hidden layer.
        On the other hand, for associating with `n+1` layers, the object activate as neurons in `virtual` visible layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        add_neuron_list = []
        cdef int i
        for i in range(neuron_count):
            feature_point_neuron = FeaturePointNeuron(VisibleNeuron())
            feature_point_neuron.activating_function = activating_function
            add_neuron_list.append(feature_point_neuron)
        self.__feature_point_neuron.append(add_neuron_list)

    def hidden_neuron_part(self, activating_function, int neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activating_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        cdef int i
        for i in range(neuron_count):
            hidden_neuron = HiddenNeuron()
            hidden_neuron.activating_function = activating_function
            self.__hidden_neuron_list.append(hidden_neuron)

    def graph_part(self, approximate_interface):
        '''
        Build complete bipartite graph.

        Args:
            approximate_interface:       The object of function approximation.
        '''
        complete_bipartite_graph = CompleteBipartiteGraph()
        complete_bipartite_graph.create_node(
            self.__visual_neuron_list,
            self.__feature_point_neuron[0]
        )
        self.__graph_list.append(complete_bipartite_graph)

        cdef int i
        for i in range(1, len(self.__feature_point_neuron)):
            complete_bipartite_graph = CompleteBipartiteGraph()
            complete_bipartite_graph.create_node(
                self.__feature_point_neuron[i - 1],
                self.__feature_point_neuron[i]
            )
            self.__graph_list.append(complete_bipartite_graph)

        complete_bipartite_graph = CompleteBipartiteGraph()
        complete_bipartite_graph.create_node(
            self.__feature_point_neuron[-1],
            self.__hidden_neuron_list
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
                ContrastiveDivergence()
            )
            self.__rbm_list.append(rbm)

        return self.__rbm_list
