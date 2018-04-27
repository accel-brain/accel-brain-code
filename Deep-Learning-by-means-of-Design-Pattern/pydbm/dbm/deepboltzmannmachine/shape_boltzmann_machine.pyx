# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
ctypedef np.float64_t DOUBLE_t


class ShapeBoltzmannMachine(DeepBoltzmannMachine):
    '''
    Shape Boltzmann Machine(Shape-BM).
    '''
    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property is read-only.")

    # auto-saved shallower visible data points which is reconstructed.
    __visible_points_arr = None

    def get_visible_points_arr(self):
        ''' getter '''
        return self.__visible_points_arr

    visible_points_arr = property(get_visible_points_arr, set_readonly)

    def __init__(
        self,
        dbm_builder,
        neuron_assign_list,
        activating_function_list,
        approximate_interface_list,
        double learning_rate,
        double dropout_rate=0.5
    ):
        '''
        Initialize deep boltzmann machine.

        Args:
            dbm_builder:            `    Concrete Builder` in Builder Pattern.
            neuron_assign_list:          The number of neurons in each layers.
            activating_function_list:    Activation function.
            approximate_interface_list:  The object of function approximation.
            learning_rate:               Learning rate.
            dropout_rate:                Dropout rate.
            inferencing_flag:            Execute inferencing or not. 
            inferencing_plan:            `each`:  Learn -> Inferece -> Learn -> ...
                                         `at_once`: All learn -> All inference   
        '''
        super().__init__(
            dbm_builder,
            neuron_assign_list,
            activating_function_list,
            approximate_interface_list,
            learning_rate,
            dropout_rate,
            inferencing_flag=True,
            inferencing_plan="each"
        )

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=1000,
        int batch_size=200,
        int r_batch_size=-1,
        sgd_flag=False
    ):
        '''
        Learning.

        Args:
            observed_data_arr:    The `np.ndarray` of observed data points.
            traning_count:        Training counts.
            batch_size:           Batch size.
            r_batch_size:         Batch size.
                                  If this value is `0`, the inferencing is a recursive learning.
                                  If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                  If this value is '-1', the inferencing is not a recursive learning.
            sgd_flag:             Learning with the stochastic gradient descent(SGD) or not.
        '''
        cdef int i
        cdef int row_i = observed_data_arr.shape[0]
        cdef int j
        cdef np.ndarray[DOUBLE_t, ndim=1] data_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] feature_point_arr
        cdef int sgd_key

        inferenced_data_list = [None] * row_i
        for k in range(traning_count):
            for i in range(row_i):
                if sgd_flag is True:
                    sgd_key = np.random.randint(row_i)
                    data_arr = observed_data_arr[sgd_key]
                else:
                    data_arr = observed_data_arr[i].copy()

                for j in range(len(self.rbm_list)):
                    self.rbm_list[j].approximate_learning(
                        data_arr,
                        1,
                        batch_size
                    )
                    feature_point_arr = self.get_feature_point(j)
                    data_arr = feature_point_arr

                rbm_list = self.rbm_list[::-1]
                for j in range(len(rbm_list)):
                    data_arr = self.get_feature_point(len(rbm_list)-1-j)
                    rbm_list[j].approximate_inferencing(
                        data_arr,
                        1,
                        r_batch_size
                    )
                if k == traning_count - 1:
                    inferenced_data_list[i] = rbm_list[-1].graph.visible_activity_arr
        self.__visible_points_arr = np.array(inferenced_data_list)

    def reshape_observed_data(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr, int overlap_n=1):
        '''
        Reshape `np.ndarray` of observed data ponints for Shape-BM.
        
        Args:
            observed_data_arr:    The `np.ndarray` of observed data points.
            overlap_n:            The number of pixels that overlaps its neighbor.

        Returns:
            np.ndarray[DOUBLE_t, ndim=2] observed data points
        '''
        cdef int row_i = observed_data_arr.shape[0]
        cdef int col_j = observed_data_arr.shape[1]
        
        if row_i != col_j:
            raise ValueError("The shape of observed data array must be sequre.")

        if row_i % 2 == 0 or col_j % 2 == 0:
            raise ValueError("The row and col of observed data array must be odd number.")

        feature_arr_list = []
        
        cdef np.ndarray[DOUBLE_t, ndim=2] target_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] feature_arr

        for i in range(int(row_i/2)):
            for j in range(int(col_j/2)):
                target_arr = observed_data_arr[i:i+overlap_n+2, j:j+overlap_n+2]

                feature_arr = np.r_[
                    target_arr[:int(target_arr.shape[0]/2), :].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2):int(target_arr.shape[0]/2)+1, :int(target_arr.shape[1]/2)].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2):int(target_arr.shape[0]/2)+1, int(target_arr.shape[1]/2):int(target_arr.shape[1]/2)+1].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2):int(target_arr.shape[0]/2)+1, int(target_arr.shape[1]/2)+1:].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2)+1:, :].reshape(-1, 1)
                ].reshape(1, -1)
                feature_arr_list.append(feature_arr)

                target_arr = target_arr.T
                feature_arr = np.r_[
                    target_arr[:int(target_arr.shape[0]/2), :].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2):int(target_arr.shape[0]/2)+1, :int(target_arr.shape[1]/2)].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2):int(target_arr.shape[0]/2)+1, int(target_arr.shape[1]/2):int(target_arr.shape[1]/2)+1].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2):int(target_arr.shape[0]/2)+1, int(target_arr.shape[1]/2)+1:].reshape(-1, 1),
                    target_arr[int(target_arr.shape[0]/2)+1:, :].reshape(-1, 1)
                ].reshape(1, -1)
                feature_arr_list.append(feature_arr)

                d_arr = target_arr[np.sort(np.diag(np.diag(target_arr)+1) == 0)]
                d_arr = np.diag(np.diag(target_arr+1)).astype(np.float64)
                d_arr[d_arr != 0] = np.inf
                d_arr = target_arr + d_arr
                d_arr = d_arr[d_arr != np.inf]
                d_key = int(d_arr.shape[0]/2)
                feature_arr = np.r_[
                    d_arr[:d_key],
                    np.diag(target_arr),
                    d_arr[d_key:]
                ].reshape(1, -1)
                feature_arr_list.append(feature_arr)

                target_arr = target_arr[::-1]
                d_arr = target_arr[np.sort(np.diag(np.diag(target_arr)+1) == 0)]
                d_arr = np.diag(np.diag(target_arr+1)).astype(np.float64)
                d_arr[d_arr != 0] = np.inf
                d_arr = target_arr + d_arr
                d_arr = d_arr[d_arr != np.inf]
                feature_arr = np.r_[
                    d_arr[:d_key],
                    np.diag(target_arr),
                    d_arr[d_key:]
                ].reshape(1, -1)
                feature_arr_list.append(feature_arr)

        cdef np.ndarray[DOUBLE_t, ndim=2] reshape_arr = np.array(feature_arr_list)[:, 0, :]
        return reshape_arr

    def reshape_inferenced_data(self, np.ndarray[DOUBLE_t, ndim=2] inferenced_data_arr):
        '''
        Reshape `np.ndarray` of inferenced data ponints for Shape-BM.
        
        Args:
            inferenced_data_arr:  The `np.ndarray` of inferenced data points.

        Returns:
            np.ndarray[DOUBLE_t, ndim=2] inferenced data points
        '''
        cdef int row_i = inferenced_data_arr.shape[0]
        cdef int col_j = inferenced_data_arr.shape[1]

        if col_j % 2 == 0:
            raise ValueError("The col of inferenced data array must be odd number.")

        mean_field_list = []
        cdef float mean_field
        for i in range(int(row_i/4)):
            mean_field = inferenced_data_arr[i*4:(i+1)*4, int(col_j/2):int(col_j/2)+1].mean()
            mean_field_list.append(mean_field)
        mean_field_arr = np.array(mean_field_list).reshape(int(row_i/4), int(row_i/4))
        shape_arr = np.zeros((int(row_i/4)+2, int(row_i/4)+2))
        shape_arr[1:1+mean_field_arr.shape[0], 1:1+mean_field_arr.shape[1]] = mean_field_arr
        return shape_arr
