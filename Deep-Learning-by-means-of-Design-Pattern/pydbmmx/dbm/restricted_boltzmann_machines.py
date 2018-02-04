# -*- coding: utf-8 -*-
from pydbm.synapse.complete_bipartite_graph import CompleteBipartiteGraph
from pydbm.approximation.interface.approximate_interface import ApproximateInterface


class RestrictedBoltzmannMachine(object):
    '''
    Restricted Boltzmann Machine.
    '''
    # Complete bipartite graph.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5
    # The object of function approximation.
    __approximate_interface = None

    def get_graph(self):
        ''' getter of graph '''
        return self.__graph

    def set_read_only(self, value):
        ''' setter of graph '''
        raise TypeError("Read Only.")

    graph = property(get_graph, set_read_only)

    def __init__(self, graph, learning_rate=0.5, approximate_interface=None):
        '''
        Initialize.

        Args:
            graph:                  Complete bipartite graph.
            learning_rate:          Learning rate.
            approximate_interface:  The object of function approximation.

        '''
        if isinstance(graph, CompleteBipartiteGraph) is False:
            raise TypeError("CompleteBipartiteGraph")

        if isinstance(approximate_interface, ApproximateInterface) is False:
            if approximate_interface is not None:
                raise TypeError("ApproximateInterface")

        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__approximate_interface = approximate_interface

    def approximate_learning(self, observed_data_arr, traning_count):
        '''
        Learning with function approximation.

        Args:
            observed_data_arr:      The array of observed data points.
            traning_count:          Training counts.
        '''

        self.__graph = self.__approximate_interface.approximate_learn(
            self.__graph,
            self.__learning_rate,
            observed_data_arr,
            traning_count=traning_count
        )

    def associate_memory(self, observed_data_arr):
        '''
        Free association with so called `Hebb ruls`.

        Args:
            observed_data_arr:   The `np.ndarray` of observed data points.
        '''
        self.__graph = self.__approximate_interface.recall(
            self.__graph,
            observed_data_arr
        )
