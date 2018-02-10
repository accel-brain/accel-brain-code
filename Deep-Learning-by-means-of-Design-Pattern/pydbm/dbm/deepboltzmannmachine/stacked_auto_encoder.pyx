# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
ctypedef np.float64_t DOUBLE_t


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
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=1000
    ):
        '''
        Learning and auto-saving featrue points with `np.ndarray`.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points.
            traning_count:          Training counts.
        '''
        cdef int row = observed_data_arr.shape[0]
        cdef int t
        cdef np.ndarray[DOUBLE_t, ndim=1] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] feature_point_arr
        feature_points_list = [None] * row
        for t in range(traning_count):
            for i in range(row):
                data_arr = observed_data_arr[i]
                super().learn(
                    observed_data_arr=np.array([data_arr]),
                    traning_count=1
                )
                if t == traning_count - 1:
                    feature_point_arr = self.get_feature_point()
                    feature_points_list[i] = feature_point_arr

        self.__feature_points_arr = np.array(feature_points_list)
