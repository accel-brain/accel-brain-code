#!/user/bin/env python
# -*- coding: utf-8 -*-
from deeplearning.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
from deeplearning.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from deeplearning.approximation.contrastive_divergence import ContrastiveDivergence
from deeplearning.activation.logistic_function import LogisticFunction
import numpy as np
import random
import pandas as pd
from pprint import pprint
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

if __name__ == "__main__":
    '''
    深層ボルツマンマシンを構成する制限ボルツマンマシンは、
    元来教師なし学習に利用されていた。
    
    これに対して、このテスト用プログラムでは二値(0/1)の目的変数を前提とした教師あり学習に応用する。
    具体的な手順としては、まず訓練（学習）時に0番目の可視層ニューロンに教師データを入力し続ける。
    同時に、1 ~ N番目の可視層ニューロンには本来の訓練データを入力する。
    そしてテスト時には、0番目に0.5の値を、1 ~ N番目にはテスト用のデータを入力する。
    するとヘッブの法則のようにボルツマンマシンが「連想」し、0 ~ N番目の活性度データを返す。
    この時、0番目の活性度を学習結果から予測された目的変数として捉えることができる。
    
    可視層ニューロンの一部やその付近に教師データを入力することで疑似的に教師あり学習を再現する試みは、
    何も新しい発想ではない。例えば以下の先行研究を参照。
    
    - Larochelle, H., & Bengio, Y. (2008, July). Classification using discriminative restricted Boltzmann machines. In Proceedings of the 25th international conference on Machine learning (pp. 536-543). ACM.
    - Larochelle, H., Mandel, M., Pascanu, R., & Bengio, Y. (2012). Learning algorithms for the classification restricted boltzmann machine. The Journal of Machine Learning Research, 13(1), 643-669.
    本プログラムは、上記の発想を単純化させたものとなる。
    '''

    # 教師データを生成する
    data_tuple = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=2,
        n_classes=2,
        class_sep=1.0,
        scale=0.1
    )
    # 説明変数と目的変数に区別する
    data_tuple_x, data_tuple_y = data_tuple
    # 訓練用のデータと検証用のデータに区別する
    traning_x, test_x, traning_y, test_y = train_test_split(
        data_tuple_x,
        data_tuple_y,
        test_size=0.3,
        random_state=888
    )

    # 深層ボルツマンマシンの引数の形式に前処理する
    traning_y = traning_y.reshape((len(traning_y), 1))
    data_arr = np.hstack((traning_y, traning_x))
    traning_data_matrix = list(data_arr)

    test_y = test_y.reshape((len(test_y), 1))
    data_arr = np.hstack((test_y, test_x))
    test_data_matrix = list(data_arr)

    evaluate_data_list = []
    
    # 深層ボルツマンマシンを構築する(第二引数の各層のニューロンの個数はデモのためアトランダムに規定）
    dbm = DeepBoltzmannMachine(
        DBMMultiLayerBuilder(),
        [len(traning_data_matrix[0]), 8, 6, 4, 2],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=1)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBMMultiLayerBuilder(),
        [len(traning_data_matrix[0]), 8, 6, 4],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=1)

    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBMMultiLayerBuilder(),
        [len(traning_data_matrix[0]), 8, 6],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=1)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBMMultiLayerBuilder(),
        [len(traning_data_matrix[0]), 8, 1],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=1)

    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    evaluate_data = pd.DataFrame(evaluate_data_list)
    evaluate_data = evaluate_data.sort_values(by=["f", "precision", "recall"], ascending=False)

    data_columns = [
        "neuron_assign_list",
        "learning_rate",
        "traning_count",
        "precision",
        "recall",
        "f"
    ]

    print(evaluate_data[data_columns])
    '''
  neuron_assign_list  learning_rate  traning_count  precision    recall  \
1      [11, 8, 6, 4]           0.05              1   0.484305  0.744828
2         [11, 8, 6]           0.05              1   0.511765  0.600000
3         [11, 8, 1]           0.05              1   0.471795  0.634483
0   [11, 8, 6, 4, 2]           0.05              1   0.459184  0.620690

          f
1  0.586957
2  0.552381
3  0.541176
0  0.527859
    '''
