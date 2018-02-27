# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
ctypedef np.float64_t DOUBLE_t


class StackedAutoEncoder(DeepBoltzmannMachine):
    '''
    Stacked Auto-Encoder.
    '''
    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property is read-only.")

    # auto-saved featrue points.
    __feature_points_arr = None

    def get_feature_points_arr(self):
        ''' getter '''
        return self.__feature_points_arr

    feature_points_arr = property(get_feature_points_arr, set_readonly)

    # auto-saved shallower visible data points which is reconstructed.
    __visible_points_arr = None

    def get_visible_points_arr(self):
        ''' getter '''
        return self.__visible_points_arr

    visible_points_arr = property(get_visible_points_arr, set_readonly)

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=1000,
        int batch_size=200,
        int r_batch_size=-1
    ):
        '''
        Learning and auto-saving featrue points with `np.ndarray`.

        Args:
            observed_data_arr:    The `np.ndarray` of observed data points.
            traning_count:        Training counts.
            batch_size:           Batch size.
            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.

        '''
        cdef int row = observed_data_arr.shape[0]
        cdef int t
        cdef np.ndarray[DOUBLE_t, ndim=1] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] feature_point_arr

        visible_points_list = [None] * row
        feature_points_list = [None] * row
        for t in range(traning_count):
            for i in range(row):
                data_arr = observed_data_arr[i]
                super().learn(
                    observed_data_arr=np.array([data_arr]),
                    traning_count=1,
                    batch_size=batch_size,
                    r_batch_size=r_batch_size
                )
                if t == traning_count - 1:
                    visible_points_arr = self.get_visible_point()
                    visible_points_list[i] = visible_points_arr
                    feature_point_arr = self.get_feature_point()
                    feature_points_list[i] = feature_point_arr

        self.__visible_points_arr = np.array(visible_points_list)
        self.__feature_points_arr = np.array(feature_points_list)
