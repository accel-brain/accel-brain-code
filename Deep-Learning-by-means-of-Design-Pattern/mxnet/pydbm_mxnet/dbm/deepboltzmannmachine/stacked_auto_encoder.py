# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
from pydbm_mxnet.dbm.deep_boltzmann_machine import DeepBoltzmannMachine


class StackedAutoEncoder(DeepBoltzmannMachine):
    '''
    Stacked Auto-Encoder.
    '''
    # auto-saved featrue points.
    __feature_points_arr = None

    def get_feature_points_arr(self):
        ''' getter '''
        return self.__feature_points_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property is read-only.")

    feature_points_arr = property(get_feature_points_arr, set_readonly)

    def learn(
        self,
        observed_data_arr,
        traning_count=1000
    ):
        '''
        Learning and auto-saving featrue points with `np.ndarray`.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points.
            traning_count:          Training counts.
        '''
        if isinstance(observed_data_arr, mx.ndarray.ndarray.NDArray) is False:
            raise TypeError()

        feature_points_list = [None] * observed_data_arr.shape[0]
        for t in range(traning_count):
            for i in range(observed_data_arr.shape[0]):
                data_arr = mx.nd.array([observed_data_arr[i].asnumpy().tolist()])
                super().learn(
                    observed_data_arr=data_arr,
                    traning_count=1
                )
                if t == traning_count - 1:
                    feature_points_arr = self.get_feature_point()
                    feature_points_list[i] = feature_points_arr.asnumpy().tolist()

        self.__feature_points_arr = mx.nd.array(feature_points_list)
