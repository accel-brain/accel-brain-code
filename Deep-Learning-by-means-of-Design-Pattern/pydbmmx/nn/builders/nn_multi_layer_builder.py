# -*- coding: utf-8 -*-
from pydbm.nn.interface.nn_builder import NNBuilder
from pydbm.synapse.neural_network_graph import NeuralNetworkGraph
import pyximport; pyximport.install()
from pydbm.neuron.visible_neuron import VisibleNeuron
from pydbm.neuron.hidden_neuron import HiddenNeuron
from pydbm.neuron.output_neuron import OutputNeuron


class NNMultiLayerBuilder(NNBuilder):
    '''
    `Concrete Builder` in Builder Pattern.
    
    Multi-layers neural networks.
    '''
    # The list of neurons in input layer.
    __input_neuron_list = []
    # The list of neurons in hidden layer.
    __hidden_neuron_list = []
    # The list of neurons in output layer.
    __output_neuron_list = []
    # The list of graphs of synapse.
    __graph_list = []

    def __init__(self):
        '''
        Initialize.
        '''
        self.__input_neuron_list = []
        self.__hidden_neuron_list = []
        self.__output_neuron_list = []
        self.__graph_list = []

    def input_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in input layer.

        Args:
            activating_function:    Activation function
            neuron_count:           The number of neurons.
        '''
        for i in range(neuron_count):
            visible_neuron = VisibleNeuron()
            visible_neuron.activating_function = activating_function
            visible_neuron.bernoulli_flag = True
            self.__input_neuron_list.append(visible_neuron)

    def hidden_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in hidden layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        add_neuron_list = []
        for i in range(neuron_count):
            hidden_neuron = HiddenNeuron()
            hidden_neuron.activating_function = activating_function
            add_neuron_list.append(hidden_neuron)
        self.__hidden_neuron_list.append(add_neuron_list)

    def output_neuron_part(self, activating_function, neuron_count):
        '''
        Build neurons in output layer.

        Args:
            activating_function:    Activation function.
            neuron_count:           The number of neurons.
        '''
        for i in range(neuron_count):
            output_neuron = OutputNeuron()
            output_neuron.activating_function = activating_function
            output_neuron.bernoulli_flag = True
            self.__output_neuron_list.append(output_neuron)

    def graph_part(self):
        '''
        Build the graphs of synapse.

        '''
        neural_network_graph = NeuralNetworkGraph(output_layer_flag=False)
        neural_network_graph.create_node(
            self.__input_neuron_list,
            self.__hidden_neuron_list[0]
        )
        self.__graph_list.append(neural_network_graph)

        for i in range(1, len(self.__hidden_neuron_list)):
            neural_network_graph = NeuralNetworkGraph(output_layer_flag=False)
            neural_network_graph.create_node(
                self.__hidden_neuron_list[i - 1],
                self.__hidden_neuron_list[i]
            )
            self.__graph_list.append(neural_network_graph)

        neural_network_graph = NeuralNetworkGraph(output_layer_flag=True)
        neural_network_graph.create_node(
            self.__hidden_neuron_list[-1],
            self.__output_neuron_list
        )
        self.__graph_list.append(neural_network_graph)

    def get_result(self):
        '''
        Return the list of builed graph of synapse.

        Returns:
            The graphs of synapse.

        '''
        return self.__graph_list
