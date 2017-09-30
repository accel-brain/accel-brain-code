#!/user/bin/env python
# -*- coding: utf-8 -*-
from multipledispatch import dispatch
from pydbm.dbm.interface.dbm_builder import DBMBuilder
from pydbm.dbm.dbm_director import DBMDirector
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.approximation.interface.approximate_interface import ApproximateInterface


class DeepBoltzmannMachine(object):
    '''
    GoFのデザイン・パタンの「Builder Pattern」のClientとして
    深層ボルツマンマシンのオブジェクトを構築する
    '''

    # 制限ボルツマンマシン
    __rbm_list = []
    # 評価時に参照する（ハイパー）パラメタの記録用辞書
    __hyper_param_dict = {}

    @dispatch(DBMBuilder, int, int, int, ActivatingFunctionInterface, ApproximateInterface, float)
    def __init__(
        self,
        dbm_builder,
        visible_neuron_count,
        feature_neuron_count,
        hidden_neuron_count,
        activating_function,
        approximate_interface,
        learning_rate
    ):
        '''
        深層ボルツマンマシンを初期化する

        Args:
            visible_neuron_count:   可視層ニューロン数
            feature_neuron_count:   特徴点の疑似可視層ニューロン数
            hidden_neuron_count:    隠れ層ニューロン数
            activating_function:    活性化関数
            approximate_interface:  近似
            learning_rate:          学習率
        '''
        dbm_builder.learning_rate = learning_rate
        dbm_director = DBMDirector(
            dbm_builder=dbm_builder
        )
        dbm_director.dbm_construct(
            neuron_assign_list=[visible_neuron_count, feature_neuron_count, hidden_neuron_count],
            activating_function=activating_function,
            approximate_interface=approximate_interface
        )

        self.__rbm_list = dbm_director.rbm_list

        self.__hyper_param_dict = {
            "visible_neuron_count": visible_neuron_count,
            "feature_neuron_count": feature_neuron_count,
            "hidden_neuron_count": hidden_neuron_count,
            "learning_rate": learning_rate,
            "activating_function": str(type(activating_function)),
            "approximate_interface": str(type(approximate_interface))
        }

    @dispatch(DBMBuilder, list, ActivatingFunctionInterface, ApproximateInterface, float)
    def __init__(
        self,
        dbm_builder,
        neuron_assign_list,
        activating_function,
        approximate_interface,
        learning_rate
    ):
        '''
        深層ボルツマンマシンを初期化する

        Args:
            neuron_assign_list:     各層のニューロンの個数 0番目が可視層で、1以上の値が隠れ層に対応する
            activating_function:    活性化関数
            approximate_interface:  近似
            learning_rate:          学習率
        '''
        dbm_builder.learning_rate = learning_rate
        dbm_director = DBMDirector(
            dbm_builder=dbm_builder
        )
        dbm_director.dbm_construct(
            neuron_assign_list=neuron_assign_list,
            activating_function=activating_function,
            approximate_interface=approximate_interface
        )
        self.__rbm_list = dbm_director.rbm_list

        self.__hyper_param_dict = {
            "neuron_assign_list": neuron_assign_list,
            "learning_rate": learning_rate,
            "activating_function": str(type(activating_function)),
            "approximate_interface": str(type(approximate_interface))
        }

    def learn(self, observed_data_matrix, traning_count=1000):
        '''
        深層ボルツマンマシンを初期化する

        Args:
            observed_data_matrix:   観測データ点リスト
            traning_count:          訓練回数
        '''
        if isinstance(observed_data_matrix, list) is False:
            raise TypeError()

        for i in range(len(self.__rbm_list)):
            rbm = self.__rbm_list[i]
            rbm.approximate_learning(observed_data_matrix, traning_count)
            feature_point_list = self.get_feature_point_list(i)
            observed_data_matrix = [feature_point_list]

        self.__hyper_param_dict["traning_count"] = traning_count

    def get_feature_point_list(self, layer_number=0):
        '''
        特徴点を抽出する

        Args:
            layer_number:   層番号（3層の場合、0なら可視層、1なら中間層、2なら隠れ層）

        Returns:
            特徴点リスト
        '''
        rbm = self.__rbm_list[layer_number]
        feature_point_list = [rbm.graph.hidden_neuron_list[j].activity for j in range(len(rbm.graph.hidden_neuron_list))]
        return feature_point_list

    def clustering(self, test_data_list):
        '''
        入力されたテストデータに関連する自由連想によりクラスタリングを実行する

        Args:
            test_data_list: 自由連想の引き金となるテストデータ

        Returns:
            自由連想結果
        '''

        rbm = self.__rbm_list[0]
        rbm.associate_memory(test_data_list)
        return self.get_visible_activity()

    def binary_clustring(self, test_data_list, exsample_key=0):
        '''
        予め生成モデルに含めておいた教師データを前提に、
        入力されたテストデータに関連する自由連想によりクラスタリングを実行する

        Args:
            test_data_list: 自由連想の引き金となるテストデータ
            exsample_key:   予め教師データを入力していた可視層ニューロンの番号

        Returns:
            自由連想結果
        '''

        clustering_list = self.clustering(test_data_list)
        if clustering_list[exsample_key] > 0.5:
            binary = 1
        else:
            binary = 0
        return (binary, clustering_list[exsample_key], test_data_list[0][1:])

    def get_visible_activity(self):
        '''
        可視層ニューロンの活性化状態を抽出する

        Returns:
            可視層ニューロンの活性化状態
        '''
        rbm = self.__rbm_list[0]
        return [rbm.graph.visible_neuron_list[i].activity for i in range(len(rbm.graph.visible_neuron_list))]

    def evaluate_bool(self, test_data_matrix):
        '''
        教師データの訓練後の予測を実行して、モデル性能をF値などの指標で算出する
        目的変数が二値(0/1)の場合の簡易版

        Args:
            test_data_matrix:   テストデータ

        Returns:
            次の辞書を返す
            {
                "tp":                 True Positive,
                "fp":                 False Positive,
                "tn":                 True Negative,
                "fn":                 False Negative,
                "precision":          適合率,
                "recall":             再現率,
                "f":                  F値,
                "<<（ハイパー）パラメタ名>>": <<値>>
            }
        '''
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        example = 0.5
        for test_data_list in test_data_matrix:
            row_bool_list = [example]
            row_bool_list.extend(test_data_list[1:])

            binary_result, value, test_data = self.binary_clustring([row_bool_list])
            if test_data_list[0] == binary_result and binary_result == 0:
                tp += 1
            elif test_data_list[0] == binary_result and binary_result == 1:
                tn += 1
            elif test_data_list[0] != binary_result and binary_result == 0:
                fp += 1
            else:
                fn += 1

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0.0
        try:
            f = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f = 0.0

        result_dict = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f": f
        }
        [result_dict.setdefault(key, val) for key, val in self.__hyper_param_dict.items()]

        return result_dict
