# -*- coding: utf-8 -*-
import mxnet as mx
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.neuron.visible_neuron import VisibleNeuron
from pydbm.neuron.hidden_neuron import HiddenNeuron
from pydbm.neuron.feature_point_neuron import FeaturePointNeuron
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.synapse.complete_bipartite_graph import CompleteBipartiteGraph
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine


class DBM3LayerBuilder(DBMBuilder):
    '''
    The `Concrete Builder` in Builder Pattern.
    
    Compose three restricted boltzmann machines for building a deep boltzmann machine.
    '''
    # The list of neurons in visible layer.
    __visual_neuron_list = []
    # The list of neurons for feature points.
    __feature_point_neuron = []
    # The list of neurons in hidden layer.
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

    def visible_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in visible layer.

        Args:
            activating_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        for i in range(neuron_count):
            visible_neuron = VisibleNeuron()
            visible_neuron.activating_function = activating_function
            visible_neuron.bernoulli_flag = True
            self.__visual_neuron_list.append(visible_neuron)

    def feature_neuron_part(self, activating_function, neuron_count):
        '''
        The object of feature points.
        Build neurons in n-layers.
        
        For associating with `n-1` layers, the object activate as neurons in hidden layer.
        On the other hand, for associating with `n+1` layers, the object activate as neurons in `virtual` visible layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        for i in range(neuron_count):
            feature_point_neuron = FeaturePointNeuron(VisibleNeuron())
            feature_point_neuron.activating_function = activating_function
            self.__feature_point_neuron.append(feature_point_neuron)

    def hidden_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
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

        self.__graph_list.append(CompleteBipartiteGraph())
        self.__graph_list.append(CompleteBipartiteGraph())

        self.__graph_list[0].create_node(
            mx.nd.array(self.__visual_neuron_list),
            mx.nd.array(self.__feature_point_neuron)
        )
        self.__graph_list[1].create_node(
            mx.nd.array(self.__feature_point_neuron),
            mx.nd.array(self.__hidden_neuron_list)
        )

    def get_result(self):
        '''
        Return the builded restricted boltzmann machines.

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
