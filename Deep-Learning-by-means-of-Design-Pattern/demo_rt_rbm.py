# -*- coding: utf-8 -*-
from pydbm.dbm.builders.rt_rbm_simple_builder import RTRBMSimpleBuilder
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
from pydbm.approximation.rt_rbm_cd import RTRBMCD
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.softmax_function import SoftmaxFunction
import numpy as np
import random
import pandas as pd
from pprint import pprint

np.seterr(all='raise')

if __name__ == "__main__":
    '''
    Thema:
        Dimention reduction
        (from 10000 * 10000 array to 10000 * 10 array)

    Detail:
        Observation data:
            The result of `np.random.uniform(size=(10000, 10000))`.

        Number of units:
            visible:                10000
            hidden(feature point):  10
            hidden:                 10000

        Activation: 
            visible:                Logistic Function
            hidden(feature point):  Logistic Function
            hidden:                 Logistic Function

        approximation: Contrastive Divergence
        learning rate: 0.05
        dropout rate:  0.5

    Result:
        TAT:
            real    1m35.472s
            user    1m32.300s
            sys     0m3.136s

        Feature points:
            0.190599  0.183594  0.482996  0.911710  0.939766  0.202852  0.042163
            0.470003  0.104970  0.602966  0.927917  0.134440  0.600353  0.264248
            0.419805  0.158642  0.328253  0.163071  0.017190  0.982587  0.779166
            0.656428  0.947666  0.409032  0.959559  0.397501  0.353150  0.614216
            0.167008  0.424654  0.204616  0.573720  0.147871  0.722278  0.068951
            .....

        Reconstruct error:
            [ 0.08297197  0.07091231  0.0823424  ...,  0.0721624   0.08404181  0.06981017]
    '''
    arr = np.arange(333, dtype=np.float64)
    arr = np.c_[
        arr+0,
        arr+1,
        arr+2,
        arr+3,
        arr+4,
        arr+5,
        arr+6,
        arr+7
    ]
    arr = np.array(arr.tolist() * 100)
    arr = arr / 333

    print(arr.shape)

    rtrbm_builder = RTRBMSimpleBuilder()
    rtrbm_builder.learning_rate = 0.00001
    rtrbm_builder.visible_neuron_part(LogisticFunction(), arr.shape[1])
    rtrbm_builder.hidden_neuron_part(LogisticFunction(), 3)
    rtrbm_builder.rnn_neuron_part(LogisticFunction())
    rtrbm_builder.graph_part(RTRBMCD())
    rbm = rtrbm_builder.get_result()

    for i in range(arr.shape[0]):
        rbm.approximate_learning(
            arr[i],
            traning_count=1, 
            batch_size=200
        )
    print(rbm.graph.visible_activity_arr)
