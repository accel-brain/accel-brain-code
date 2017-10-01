#!/user/bin/env python
# -*- coding: utf-8 -*-
from pydbm.synapse.complete_bipartite_graph import CompleteBipartiteGraph
from pydbm.approximation.interface.approximate_interface import ApproximateInterface


class RestrictedBoltzmannMachine(object):
    '''
    制限ボルツマンマシン
    '''
    # 完全二部グラフ
    __graph = None
    # 学習率（係数）
    __learning_rate = 0.5
    # 学習の近似用オブジェクト
    __approximate_interface = None

    def get_graph(self):
        ''' getter of graph '''
        return self.__graph

    def set_read_only(self, value):
        ''' setter of graph '''
        raise TypeError("Read Only.")

    graph = property(get_graph, set_read_only)

    def __init__(self, graph, double learning_rate=0.5, approximate_interface=None):
        '''
        初期化

        Args:
            graph:                  完全二部グラフ
            learning_rate:          学習率
            approximate_interface:  近似用オブジェクト

        '''
        if isinstance(graph, CompleteBipartiteGraph) is False:
            raise TypeError("CompleteBipartiteGraph")

        if isinstance(approximate_interface, ApproximateInterface) is False:
            if approximate_interface is not None:
                raise TypeError("ApproximateInterface")

        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__approximate_interface = approximate_interface

    def approximate_learning(self, observed_data_matrix, int traning_count):
        '''
        近似による学習を実行する

        Args:
            observed_data_matrix:   観測データ点リスト
            traning_count:          訓練回数
        '''

        self.__graph = self.__approximate_interface.approximate_learn(
            self.__graph,
            self.__learning_rate,
            observed_data_matrix,
            traning_count=traning_count
        )

    def associate_memory(self, observed_data_matrix):
        '''
        記憶から自由連想する
        ヘブ則的発想

        Args:
            observed_data_matrix:   観測データ点リスト
        '''
        self.__graph = self.__approximate_interface.recall(
            self.__graph,
            observed_data_matrix
        )
