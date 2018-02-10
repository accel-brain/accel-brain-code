# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
import random
from pydbm_mxnet.dbm.deepboltzmannmachine.stacked_auto_encoder import StackedAutoEncoder
from pydbm_mxnet.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbm_mxnet.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm_mxnet.activation.logistic_function import LogisticFunction
from pydbm_mxnet.activation.relu_function import ReLuFunction


if __name__ == "__main__":
    '''
    '''
    target_arr = mx.ndarray.random_poisson(shape=(10000, 1000), dtype="float32")

    dbm = StackedAutoEncoder(
        DBMMultiLayerBuilder(),
        [target_arr.shape[1], 10, target_arr.shape[1]],
        [ReLuFunction(), LogisticFunction(), ReLuFunction()],
        ContrastiveDivergence(),
        0.0005,
        0.25
    )
    dbm.learn(target_arr, traning_count=1)
    print(dbm.feature_points_arr)
