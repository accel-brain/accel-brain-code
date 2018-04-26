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
