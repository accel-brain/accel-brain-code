#!/user/bin/env python
# -*- coding: utf-8 -*-
import sys

from deeplearning.dbm.interface.dbm_builder import DBMBuilder
from deeplearning.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
from deeplearning.activation.interface.activating_function_interface import ActivatingFunctionInterface
from deeplearning.approximation.interface.approximate_interface import ApproximateInterface


class DBMDirector(object):
    '''
    GoFのデザイン・パタンの「Builder Pattern」の「監督者」
    制限ボルツマンマシンを組み立てることで、
    深層ボルツマンマシンのオブジェクトを生成する
    '''

    # GoFのデザイン・パタンの「Bulder Pattern」の「建築者」
    __dbm_builder = None
    # 制限ボルツマンマシンのリスト
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
        「建築者」を初期化する
        
        Args:
            dbm_builder     Builder Patternの「具体的な建築者」
        '''
        if isinstance(dbm_builder, DBMBuilder) is False:
            raise TypeError()

        self.__dbm_builder = dbm_builder

    def dbm_construct(
        self,
        visible_neuron_count,
        feature_neuron_count,
        hidden_neuron_count,
        activating_function,
        approximate_interface
    ):
        '''
        深層ボルツマンマシンを構築する

        Args:
            visible_neuron_count:   可視層ニューロン数
            feature_neuron_count:   特徴点の疑似可視層ニューロン数
            hidden_neuron_count:    隠れ層ニューロン数
            activating_function:    活性化関数
            approximate_interface:  近似
        '''
        if isinstance(activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()

        if isinstance(approximate_interface, ApproximateInterface) is False:
            raise TypeError()

        self.__dbm_builder.visible_neuron_part(activating_function, visible_neuron_count)
        self.__dbm_builder.feature_neuron_part(activating_function, feature_neuron_count)
        self.__dbm_builder.hidden_neuron_part(activating_function, hidden_neuron_count)
        self.__dbm_builder.graph_part(approximate_interface)
        self.rbm_list = self.__dbm_builder.get_result()
