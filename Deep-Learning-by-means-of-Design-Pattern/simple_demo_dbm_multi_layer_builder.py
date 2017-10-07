#!/user/bin/env python
# -*- coding: utf-8 -*-
import pyximport; pyximport.install()
from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.activation.logistic_function import LogisticFunction
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import random
import pandas as pd
from pprint import pprint
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

if __name__ == "__main__":
    '''
    '''

    # 教師データを生成する
    data_tuple = make_classification(
        n_samples=10000,
        n_features=100,
        n_informative=5,
        n_classes=5,
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

    print(len(traning_data_matrix[0]))
    
    # 深層ボルツマンマシンを構築する(第二引数の各層のニューロンの個数はデモのためアトランダムに規定）
    dbm = DeepBoltzmannMachine(
        DBMMultiLayerBuilder(),
        [len(traning_data_matrix[0]), 10, len(traning_data_matrix[0])],
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
