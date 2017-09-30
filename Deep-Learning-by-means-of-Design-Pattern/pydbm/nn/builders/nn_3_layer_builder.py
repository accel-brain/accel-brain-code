#!/user/bin/env python
# -*- coding: utf-8 -*-
from pydbm.nn.interface.nn_builder import NNBuilder
from pydbm.neuron.visible_neuron import VisibleNeuron
from pydbm.neuron.hidden_neuron import HiddenNeuron
from pydbm.neuron.output_neuron import OutputNeuron
from pydbm.synapse.neural_network_graph import NeuralNetworkGraph


class NN3LayerBuilder(NNBuilder):
    '''
    GoFのデザイン・パタンの「Builder Pattern」の「具体的建築者」
    3層のニューラルネットワークのオブジェクトを生成する
    '''
    # 可視層ニューロンのリスト
    __input_neuron_list = []
    # 隠れ層ニューロンのリスト
    __hidden_neuron_list = []
    # 出力層ニューロンのリスト
    __output_neuron_list = []
    # グラフ
    __graph_list = []

    def __init__(self):
        '''
        初期化する
        '''
        self.__input_neuron_list = []
        self.__hidden_neuron_list = []
        self.__output_neuron_list = []
        self.__graph_list = []

    def input_neuron_part(self, activating_function, neuron_count):
        '''
        インターフェイスの実現
        可視層ニューロンを構築する

        Args:
            activating_function:    活性化関数
            neuron_count:           ニューロン数
        '''
        for i in range(neuron_count):
            visible_neuron = VisibleNeuron()
            visible_neuron.activating_function = activating_function
            visible_neuron.bernoulli_flag = True
            self.__input_neuron_list.append(visible_neuron)

    def hidden_neuron_part(self, activating_function, neuron_count):
        '''
        インターフェイスの実現
        中間層ニューロンを構築する

        Args:
            activating_function:    活性化関数
            neuron_count:           ニューロン数
        '''

        for i in range(neuron_count):
            hidden_neuron = HiddenNeuron()
            hidden_neuron.activating_function = activating_function
            self.__hidden_neuron_list.append(hidden_neuron)

    def output_neuron_part(self, activating_function, neuron_count):
        '''
        インターフェイスの実現
        出力層ニューロンを構築する

        Args:
            activating_function:    活性化関数
            neuron_count:           ニューロン数
        '''

        for i in range(neuron_count):
            output_neuron = OutputNeuron()
            output_neuron.activating_function = activating_function
            output_neuron.bernoulli_flag = True
            self.__output_neuron_list.append(output_neuron)

    def graph_part(self):
        '''
        インターフェイスの実現
        ニューラルネットワークのグラフを構築する

        '''
        neural_network_graph = NeuralNetworkGraph(output_layer_flag=False)
        neural_network_graph.create_node(
            self.__input_neuron_list,
            self.__hidden_neuron_list
        )
        self.__graph_list.append(neural_network_graph)

        neural_network_graph = NeuralNetworkGraph(output_layer_flag=True)
        neural_network_graph.create_node(
            self.__hidden_neuron_list,
            self.__output_neuron_list
        )
        self.__graph_list.append(neural_network_graph)

    def get_result(self):
        '''
        インターフェイスの実現
        構築したニューラルネットワークのリストを返す

        Returns:
            ニューラルネットワークのオブジェクトのリスト

        '''
        return self.__graph_list
