#!/user/bin/env python
# -*- coding: utf-8 -*-

from deeplearning.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
from deeplearning.dbm.builders.dbm_3_layer_builder import DBM3LayerBuilder
from deeplearning.approximation.contrastive_divergence import ContrastiveDivergence
from deeplearning.activation.sigmoid_function import SigmoidFunction
from sklearn.datasets import load_iris
import numpy as np
import random
import pandas as pd
from pprint import pprint


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

    # irisのデータセットを参照する
    data_arr = load_iris()
    feature_arr = data_arr["data"]
    target_arr = data_arr["target"]

    target_arr = target_arr.reshape((len(target_arr), 1))
    data_arr = np.hstack((target_arr, feature_arr))

    # 目的変数を0-1に限定する
    data_matrix = list(data_arr[:99])

    # シャッフルし、訓練用データとテスト用データに分割する
    random.shuffle(data_matrix)
    traning_data_matrix = data_matrix[:69]
    test_data_matrix = data_matrix[70:99]

    evaluate_data_list = []

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )

    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)
    evaluate_data = pd.DataFrame(evaluate_data_list)
    evaluate_data = evaluate_data.sort_values(by=["f", "precision", "recall"], ascending=False)

    data_columns = [
        "visible_neuron_count",
        "feature_neuron_count",
        "hidden_neuron_count",
        "learning_rate",
        "traning_count",
        "precision",
        "recall",
        "f"
    ]

    print(evaluate_data[data_columns])
    '''
   visible_neuron_count  feature_neuron_count  hidden_neuron_count  \
4                     5                    10                    5
1                     5                     3                    2
7                     5                     3                    2
0                     5                    10                    5
3                     5                     3                    2
6                     5                    10                    5
2                     5                    10                    5
5                     5                     3                    2

   learning_rate  traning_count  precision    recall         f
4           0.05            100   0.555556  0.769231  0.645161
1           0.05              2   0.526316  0.769231  0.625000
7           0.50            100   0.538462  0.538462  0.538462
0           0.05              2   0.421053  0.615385  0.500000
3           0.50              2   0.421053  0.615385  0.500000
6           0.50            100   0.461538  0.461538  0.461538
2           0.50              2   0.428571  0.461538  0.444444
5           0.05            100   0.333333  0.307692  0.320000 
    '''
