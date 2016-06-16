#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from deeplearning.activation.interface.activating_function_interface import ActivatingFunctionInterface


class Neuron(metaclass=ABCMeta):
    '''
    ニューラルネットワークのニューロンに関する抽象クラス
    GoFのデザイン・パタンにおける「Template Method Pattern」の構成
    '''

    # バイアス
    __bias = 0.0
    # バイアス差分
    __diff_bias = 0.0
    # 活性化関数
    __activating_function = None
    # 活性度
    __activity = 0.0

    def get_bias(self):
        ''' getter of bias '''
        if isinstance(self.__bias, float) is False:
            raise TypeError()
        return self.__bias

    def set_bias(self, value):
        ''' setter of bias '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__bias = value

    def get_diff_bias(self):
        ''' getter of diff_bias '''
        if isinstance(self.__diff_bias, float) is False:
            raise TypeError()
        return self.__diff_bias

    def set_diff_bias(self, value):
        ''' setter of diff_bias '''
        if isinstance(value, float) is False:
            raise TypeError()
        self.__diff_bias = value

    def get_activating_function(self):
        ''' getter of activating_function '''
        if isinstance(self.__activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.__activating_function

    def set_activating_function(self, value):
        ''' setter of activating_function '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.__activating_function = value

    def get_activity(self):
        ''' getter of activity'''
        if isinstance(self.__activity, float) is False:
            raise TypeError()
        return self.__activity

    def set_activity(self, value):
        ''' setter of activity '''
        if isinstance(value, float) is False and isinstance(value, int) is False:
            raise TypeError()
        self.__activity = float(value)

    bias = property(get_bias, set_bias)
    diff_bias = property(get_diff_bias, set_diff_bias)
    activating_function = property(get_activating_function, set_activating_function)
    activity = property(get_activity, set_activity)

    def activate(self, link_value):
        '''
        活性化させる

        Args:
            link_value    入力値

        Returns:
            true => 活性化 false => 非活性化
        '''
        output = self.activating_function.activate(
            link_value + self.bias
        )
        return output

    @abstractmethod
    def update_bias(self, learning_rate):
        '''
        バイアス差分を更新する

        Args:
            learning_rate:  学習率
        '''
        raise NotImplementedError()

    def learn_bias(self):
        '''
        バイアスの学習を実行する
        '''
        self.bias += self.diff_bias
        self.diff_bias = 0.0
