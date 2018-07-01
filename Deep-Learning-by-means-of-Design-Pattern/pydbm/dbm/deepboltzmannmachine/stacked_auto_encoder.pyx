# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
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
        int traning_count=-1,
        int batch_size=200,
        int r_batch_size=-1,
        sgd_flag=False,
        int training_count=1000
    ):
        '''
        Learning and auto-saving featrue points with `np.ndarray`.

        Args:
            observed_data_arr:    The `np.ndarray` of observed data points.
            training_count:       Training counts.
            batch_size:           Batch size.
            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.
            sgd_flag:             Learning with the stochastic gradient descent(SGD) or not.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        cdef int row_i = observed_data_arr.shape[0]
        cdef int t
        cdef np.ndarray[DOUBLE_t, ndim=1] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] feature_point_arr
        cdef int sgd_key

        visible_points_list = [None] * row_i
        feature_points_list = [None] * row_i
        for t in range(training_count):
            for i in range(row_i):
                if t == traning_count - 1:
                    data_arr = observed_data_arr[i]
                else:
                    if sgd_flag is True:
                        sgd_key = np.random.randint(row_i)
                        data_arr = observed_data_arr[sgd_key]
                    else:
                        data_arr = observed_data_arr[i]

                super().learn(
                    observed_data_arr=np.array([data_arr]),
                    training_count=1,
                    batch_size=batch_size,
                    r_batch_size=r_batch_size,
                    sgd_flag=False
                )
                if t == training_count - 1:
                    visible_points_arr = self.get_visible_point()
                    visible_points_list[i] = visible_points_arr
                    feature_point_arr = self.get_feature_point()
                    feature_points_list[i] = feature_point_arr

        self.__visible_points_arr = np.array(visible_points_list)
        self.__feature_points_arr = np.array(feature_points_list)
