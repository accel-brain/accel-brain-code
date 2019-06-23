# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
ctypedef np.float64_t DOUBLE_t


class StackedAutoEncoder(DeepBoltzmannMachine):
    '''
    Stacked Auto-Encoder.
    
    DBM is functionally equivalent to a Stacked Auto-Encoder, 
    which is-a neural network that tries to reconstruct its input. 
    To encode the observed data points, the function of DBM is as 
    linear transformation of feature map. On the other hand, 
    to decode this feature points, the function of DBM is as 
    linear transformation of feature map.
    
    The reconstruction error should be calculated in relation to problem setting. 
    This library provides a default method, which can be overridden, for error 
    function that computes Mean Squared Error(MSE).

    References:
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_stacked_auto_encoder.ipynb
        - Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive science, 9(1), 147-169.
        - Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
        - Le Roux, N., & Bengio, Y. (2008). Representational power of restricted Boltzmann machines and deep belief networks. Neural computation, 20(6), 1631-1649.
        - Salakhutdinov, R., & Hinton, G. E. (2009). Deep boltzmann machines. InInternational conference on artificial intelligence and statistics (pp. 448-455).
    
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
        sgd_flag=None,
        int training_count=1000
    ):
        '''
        Learning and auto-saving featrue points with `np.ndarray`.

        Args:
            observed_data_arr:      The `np.ndarray` of observed data points.
            training_count:         Training counts.
            batch_size:             Batch size.
            r_batch_size:           Batch size.
                                    If this value is `0`, the inferencing is a recursive learning.
                                    If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                    If this value is '-1', the inferencing is not a recursive learning.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        super().learn(
            observed_data_arr=observed_data_arr,
            training_count=training_count,
            batch_size=batch_size,
            r_batch_size=r_batch_size
        )

        cdef int start_index
        cdef int end_index
        cdef np.ndarray[DOUBLE_t, ndim=2] visible_points_arr = None
        cdef np.ndarray[DOUBLE_t, ndim=2] feature_points_arr = None
        if batch_size > 0:
            for i in range(int(observed_data_arr.shape[0] / batch_size)):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                super().learn(
                    observed_data_arr=observed_data_arr[start_index:end_index],
                    training_count=1,
                    batch_size=0,
                    r_batch_size=0
                )
                if visible_points_arr is None:
                    visible_points_arr = self.get_visible_point()
                else:
                    visible_points_arr = np.r_[visible_points_arr, self.get_visible_point()]

                if feature_points_arr is None:
                    feature_points_arr = self.get_feature_point()
                else:
                    feature_points_arr = np.r_[feature_points_arr, self.get_feature_point()]

        else:
            visible_points_arr = self.get_visible_point()
            feature_points_arr = self.get_feature_point()

        self.__visible_points_arr = visible_points_arr
        self.__feature_points_arr = feature_points_arr
