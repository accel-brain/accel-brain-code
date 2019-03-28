# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class BatchNorm(object):
    '''
    Batch normalization for a regularization.

    References:
        - Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.
    '''

    # Inferencing and test mode or not.
    __test_mode = False

    def get_test_mode(self):
        ''' getter '''
        return self.__test_mode
    
    def set_test_mode(self, value):
        ''' setter '''
        self.__test_mode = value

    test_mode = property(get_test_mode, set_test_mode)

    # `np.ndarray` of delta of beta.
    __beta_arr = None

    # `np.ndarray` of delta of gamma.
    __gamma_arr = None

    # `np.ndarray` of delta of beta.
    __delta_beta_arr = None

    # `np.ndarray` of delta of gamma.
    __delta_gamma_arr = None

    def __init__(
        self, 
        momentum=0.89, 
        init_test_mean_arr=None,
        init_test_var_arr=None
    ):
        '''
        Init.

        Args:
            momentum:               Momentum.
            init_test_mean_arr:     `np.ndarray` of means in inferencing and test mode.
            init_test_var_arr:      `np.ndarray` of variances in inferencing and test mode
        '''
        self.test_mode = False
        self.__momentum = momentum

        self.__beta_arr = None
        self.__gamma_arr = None

        self.__init_test_mean_arr = init_test_mean_arr
        self.__init_test_var_arr = init_test_var_arr

        self.__test_mean_arr = None
        self.__test_var_arr = None

        self.__std_arr = None
        self.__z_scored_arr = None
        self.__mean_diff_arr = None

    def forward_propagation(self, np.ndarray observed_arr):
        '''
        Forward propagation.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Retunrs:
            `np.ndarray` of normalized data.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=3] _observed_arr
        observed_shape = observed_arr.copy().shape
        cdef int batch_size = observed_shape[0]
        cdef int seq_len = 1
        if observed_arr.ndim > 2:
            seq_len = observed_arr.shape[1]

        if observed_arr.ndim > 2:
            _observed_arr = observed_arr.reshape((batch_size, seq_len, -1))
        else:
            _observed_arr = observed_arr.reshape((batch_size, 1, -1))

        cdef np.ndarray[DOUBLE_t, ndim=1] mu_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] var_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] std_arr = np.empty((seq_len, _observed_arr[0].shape[1]))
        cdef np.ndarray[DOUBLE_t, ndim=3] mean_diff_arr = np.empty_like(_observed_arr)
        cdef np.ndarray[DOUBLE_t, ndim=3] z_scored_arr = np.empty_like(_observed_arr)
        cdef np.ndarray[DOUBLE_t, ndim=3] test_mean_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] test_var_arr

        if self.__init_test_mean_arr is None:
            test_mean_arr = np.zeros_like(_observed_arr)
        else:
            test_mean_arr = self.__init_test_mean_arr

        if self.__init_test_var_arr is None:
            test_var_arr = np.zeros_like(_observed_arr)
        else:
            test_var_arr = self.__init_test_var_arr

        if self.test_mode is False:
            for seq in range(seq_len):
                mu_arr = _observed_arr[:, seq].mean(axis=0)
                mean_diff_arr[:, seq] = _observed_arr[:, seq] - mu_arr
                var_arr = np.power(mean_diff_arr[:, seq], 2).mean(axis=0)
                std_arr[seq] = np.sqrt(var_arr + 1e-08)
                z_scored_arr[:, seq] = mean_diff_arr[:, seq] / std_arr[seq]
                test_mean_arr[:, seq] = self.__momentum * test_mean_arr[:, seq] + (1 - self.__momentum) * mu_arr
                test_var_arr[:, seq] = self.__momentum * test_var_arr[:, seq] + (1 - self.__momentum) * var_arr
        else:
            mean_diff_arr = _observed_arr - self.__test_mean_arr
            z_scored_arr = mean_diff_arr / np.sqrt(self.__test_var_arr + 1e-08)

        self.__std_arr = std_arr
        self.__mean_diff_arr = mean_diff_arr
        self.__z_scored_arr = z_scored_arr
        self.__test_mean_arr = test_mean_arr
        self.__test_var_arr = test_var_arr

        if self.__beta_arr is None:
            self.__beta_arr = np.zeros_like(z_scored_arr)
        if self.__gamma_arr is None:
            self.__gamma_arr = np.ones_like(z_scored_arr)

        return (self.__gamma_arr * z_scored_arr + self.__beta_arr).reshape(observed_shape)

    def back_propagation(self, np.ndarray delta_arr):
        '''
        Back propagation.

        Args:
            delta_arr:  `np.ndarray` of delta.
        
        Returns:
            `np.ndarray` of delta.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=3] _delta_arr
        delta_shape = delta_arr.copy().shape

        cdef int batch_size = delta_shape[0]
        cdef int seq_len = 1
        if delta_arr.ndim > 2:
            seq_len = delta_arr.shape[1]

        if delta_arr.ndim > 2:
            _delta_arr = delta_arr.reshape((batch_size, seq_len, -1))
        else:
            _delta_arr = delta_arr.reshape((batch_size, 1, -1))

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_beta_arr = delta_arr.reshape((batch_size, -1))
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_gamma_arr = self.__z_scored_arr.reshape((batch_size, -1)) * delta_arr.reshape((batch_size, -1))
        self.__delta_beta_arr = delta_beta_arr
        self.__delta_gamma_arr = delta_gamma_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_z_score_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_mean_diff_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_std_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_var_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_mean_arr

        for seq in reversed(range(seq_len)):
            delta_z_score_arr = self.__gamma_arr[:, seq] * _delta_arr[:, seq]
            delta_mean_diff_arr = delta_z_score_arr / self.__std_arr[seq]
            delta_std_arr = -np.sum((delta_z_score_arr * self.__mean_diff_arr[:, seq]) / (np.power(self.__std_arr[seq], 2)), axis=0)
            delta_var_arr = (1/2) * delta_std_arr / self.__std_arr[seq]
            delta_mean_diff_arr += (2 / batch_size) * self.__mean_diff_arr[:, seq] * delta_var_arr
            delta_mean_arr = delta_mean_diff_arr.sum(axis=0)
            _delta_arr[:, seq] = delta_mean_diff_arr - delta_mean_arr / batch_size

        delta_arr = _delta_arr.reshape(delta_shape)
        return delta_arr

    def get_beta_arr(self):
        ''' getter '''
        return self.__beta_arr

    def set_beta_arr(self, value):
        ''' setter '''
        self.__beta_arr = value

    beta_arr = property(get_beta_arr, set_beta_arr)

    def get_gamma_arr(self):
        ''' getter '''
        return self.__gamma_arr

    def set_gamma_arr(self, value):
        ''' setter '''
        self.__gamma_arr = value

    gamma_arr = property(get_gamma_arr, set_gamma_arr)

    def get_delta_beta_arr(self):
        ''' getter '''
        return self.__delta_beta_arr
    
    def set_delta_beta_arr(self, value):
        ''' setter '''
        self.__delta_beta_arr = value
    
    delta_beta_arr = property(get_delta_beta_arr, set_delta_beta_arr)

    def get_delta_gamma_arr(self):
        ''' getter '''
        return self.__delta_gamma_arr
    
    def set_delta_gamma_arr(self, value):
        ''' setter '''
        self.__delta_gamma_arr = value
    
    delta_gamma_arr = property(get_delta_gamma_arr, set_delta_gamma_arr)
