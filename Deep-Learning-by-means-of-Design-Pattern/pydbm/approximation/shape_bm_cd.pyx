# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
cimport cython
from pydbm.optimization.opt_params import OptParams
from pydbm.optimization.optparams.sgd import SGD
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
ctypedef np.float64_t DOUBLE_t


class ShapeBMCD(ApproximateInterface):
    '''
    Contrastive Divergence for Shape-Boltzmann machine(Shape-BM).
    
    Conceptually, the positive phase is to the negative phase what waking is to sleeping.

    The concept of Shape Boltzmann Machine (Eslami, S. A., et al. 2014) 
    provided inspiration to this library.
    
    The usecases of Shape-BM are image segmentation, object detection, inpainting and graphics. Shape-BM is the model for the task of modeling binary shape images, in that samples from the model look realistic and it can generalize to generate samples that differ from training examples.

    References:
        - Eslami, S. A., Heess, N., Williams, C. K., & Winn, J. (2014). The shape boltzmann machine: a strong model of object shape. International Journal of Computer Vision, 107(2), 155-176.

    '''

    # The list of the reconstruction error rate (MSE)
    __reconstruct_error_list = []

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This is read-only.")

    def get_reconstrct_error_list(self):
        ''' getter '''
        return self.__reconstruct_error_list

    reconstruct_error_list = property(get_reconstrct_error_list, set_readonly)

    # Graph of neurons.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5
    # Dropout rate.
    __dropout_rate = 0.5
    # Batch size in learning.
    __batch_size = 0
    # Batch step in learning.
    __batch_step = 0
    # Batch size in inference(recursive learning or not).
    __r_batch_size = 0
    # Batch step in inference(recursive learning or not).
    __r_batch_step = 0
    
    # the pair of layers.
    __v_h_flag = None
    # The number of overlaped pixels.
    __overlap_n = 1
    
    def __init__(self, v_h_flag, overlap_n=1, opt_params=None):
        '''
        Init.
        
        Args:
            v_h_flag:       If this value is `True`, the pair of layers is visible layer and hidden layer.
                            If this value is `False`, the pair of layers is hidden layer and hidden layer.
            overlap_n:      The number of overlapping.
            opt_params:     Optimization function.

        '''
        if isinstance(v_h_flag, bool):
            self.__v_h_flag = v_h_flag
        else:
            raise TypeError()

        if isinstance(overlap_n, int):
            self.__overlap_n = overlap_n
        else:
            raise TypeError()

        if opt_params is None:
            opt_params = SGD(momentum=0.0)

        if isinstance(opt_params, OptParams):
            self.__opt_params = opt_params
        else:
            raise TypeError()

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=-1,
        int batch_size=200,
        int training_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                          Graph of neurons.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            observed_data_arr:              Observed data points.
            training_count:                 Training counts.
            batch_size:                     Batch size (0: not mini-batch)

        Returns:
            Graph of neurons.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size

        cdef int epoch
        for epoch in range(training_count):
            if ((epoch + 1) % attenuate_epoch == 0):
                self.__learning_rate = self.__learning_rate * learning_attenuate_rate

            if self.__batch_size > 0:
                rand_index = np.random.choice(observed_data_arr.shape[0], size=self.__batch_size)
                if self.__v_h_flag is True:
                    self.__v_h_learn(observed_data_arr[rand_index])
                else:
                    self.__h_h_learn(observed_data_arr[rand_index])
            else:
                if self.__v_h_flag is True:
                    self.__v_h_learn(observed_data_arr)
                else:
                    self.__h_h_learn(observed_data_arr)

        return self.__graph

    def approximate_inference(
        self,
        graph,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        int traning_count=-1,
        int r_batch_size=200,
        int training_count=1000
    ):
        '''
        Inference with function approximation.

        Args:
            graph:                          Graph of neurons.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            observed_data_arr:              Observed data points.
            training_count:                 Training counts.
            r_batch_size:                   Batch size.
                                            If this value is `0`, the inferencing is a recursive learning.
                                            If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                            If this value is '-1', the inferencing is not a recursive learning.

        Returns:
            Graph of neurons.
        '''
        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.__graph = graph
        self.__learning_rate = learning_rate
        self.__r_batch_size = r_batch_size

        if self.__r_batch_size != self.__batch_size:
            batch_size = self.__batch_size
        else:
            batch_size = self.__r_batch_size

        cdef int epoch
        for epoch in range(training_count):
            if ((epoch + 1) % attenuate_epoch == 0):
                self.__learning_rate = self.__learning_rate * learning_attenuate_rate

            if batch_size > 0:
                rand_index = np.random.choice(observed_data_arr.shape[0], size=batch_size)
                if self.__v_h_flag is True:
                    self.__v_h_inference(observed_data_arr[rand_index])
                else:
                    self.__h_h_inference(observed_data_arr[rand_index])
            else:
                if self.__v_h_flag is True:
                    self.__v_h_inference(observed_data_arr)
                else:
                    self.__h_h_inference(observed_data_arr)

        return self.__graph

    def __v_h_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        Args:
            observed_data_list:      observed data points.
        '''
        # Waking.
        self.__graph.visible_activity_arr = observed_data_arr.copy()

        cdef int split_v_num = int((self.__graph.visible_activity_arr.shape[1] - self.__overlap_n) / 2)
        cdef int split_h_num = int(self.__graph.weights_arr.shape[1] / 2)

        cdef np.ndarray[DOUBLE_t, ndim=2] left_hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] right_hidden_activity_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] left_visible_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] center_visible_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] right_visible_activity_arr

        cdef np.ndarray[DOUBLE_t, ndim=1] center_visible_sum_arr

        if self.__graph.visible_activity_arr.shape[1] % 2 != 0:
            left_hidden_activity_arr = np.dot(
                self.__graph.visible_activity_arr[:, :split_v_num],
                self.__graph.weights_arr[:split_v_num, :split_h_num]
            ) + self.__graph.hidden_bias_arr[:split_h_num]
            
            right_hidden_activity_arr = np.dot(
                self.__graph.visible_activity_arr[:, -split_v_num:],
                self.__graph.weights_arr[-split_v_num:, split_h_num:]
            ) + self.__graph.hidden_bias_arr[-split_h_num:]

            # Overlapping for Shape-BM.
            self.__graph.hidden_activity_arr = np.c_[
                left_hidden_activity_arr, 
                right_hidden_activity_arr
            ]
            self.__graph.hidden_activity_arr = np.nan_to_num(self.__graph.hidden_activity_arr)
        else:
            self.__graph.hidden_activity_arr = np.dot(
                self.__graph.visible_activity_arr,
                self.__graph.weights_arr
            ) + self.__graph.hidden_bias_arr

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            self.__graph.hidden_activity_arr
        )

        self.__graph.hidden_activity_arr = self.__opt_params.dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr += np.dot(
            self.__graph.visible_activity_arr.T,
            self.__graph.hidden_activity_arr
        )

        self.__graph.visible_diff_bias_arr += np.nansum(self.__graph.visible_activity_arr, axis=0)
        self.__graph.hidden_diff_bias_arr += np.nansum(self.__graph.hidden_activity_arr, axis=0)

        # Sleeping.
        split_num = int(self.__graph.hidden_activity_arr.shape[1] / 2)
        
        cdef np.ndarray[DOUBLE_t, ndim=2] center_link_value_arr

        if self.__graph.hidden_activity_arr.shape[1] % 2 == 0:
            left_visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr[:, :split_h_num],
                self.__graph.weights_arr.T[:split_h_num, :split_v_num]
            ) + self.__graph.visible_bias_arr[:split_v_num]
            
            center_visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr,
                self.__graph.weights_arr.T
            ) + self.__graph.visible_bias_arr
            
            right_visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr[:, -split_h_num:],
                self.__graph.weights_arr.T[split_h_num:, -split_v_num:]
            ) + self.__graph.visible_bias_arr[-split_v_num:]
        
            # Overlapping for Shape-BM.
            if left_visible_activity_arr.shape[1] + self.__overlap_n + right_visible_activity_arr.shape[1] < self.__graph.visible_activity_arr.shape[1]:
                center_n = self.__overlap_n + 1
            else:
                center_n = self.__overlap_n

            center_visible_sum_arr = np.nansum(center_visible_activity_arr, axis=1)
            self.__graph.visible_activity_arr = np.c_[
                left_visible_activity_arr,
                np.array(
                    [
                        center_visible_sum_arr.reshape(-1, 1)
                    ] * center_n
                ).reshape((center_visible_sum_arr.shape[0], -1)),
                right_visible_activity_arr
            ]
            self.__graph.visible_activity_arr = np.nan_to_num(self.__graph.visible_activity_arr)
        else:
            self.__graph.visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr,
                self.__graph.weights_arr
            ) + self.__graph.visible_bias_arr

        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(
            self.__graph.visible_activity_arr
        )

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            np.dot(
                self.__graph.visible_activity_arr,
                self.__graph.weights_arr
            ) + self.__graph.hidden_bias_arr
        )

        self.__graph.hidden_activity_arr = self.__opt_params.de_dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr -= np.dot(
            self.__graph.visible_activity_arr.T,
            self.__graph.hidden_activity_arr
        )

        self.__graph.visible_diff_bias_arr -= np.nansum(self.__graph.visible_activity_arr, axis=0)
        self.__graph.hidden_diff_bias_arr -= np.nansum(self.__graph.hidden_activity_arr, axis=0)

        self.__graph.diff_weights_arr += self.__opt_params.compute_weight_decay_delta(
            self.__graph.weights_arr
        )

        # Learning.
        params_list= [
            self.__graph.visible_bias_arr,
            self.__graph.hidden_bias_arr,
            self.__graph.weights_arr
        ]
        grads_list = [
            self.__graph.visible_diff_bias_arr,
            self.__graph.hidden_diff_bias_arr,
            self.__graph.diff_weights_arr
        ]

        if self.__graph.visible_activating_function.batch_norm is not None:
            params_list.append(
                self.__graph.visible_activating_function.batch_norm.beta_arr
            )
            params_list.append(
                self.__graph.visible_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.__graph.visible_activating_function.batch_norm.delta_beta_arr
            )
            grads_list.append(
                self.__graph.visible_activating_function.batch_norm.delta_gamma_arr
            )

        if self.__graph.hidden_activating_function.batch_norm is not None:
            params_list.append(
                self.__graph.hidden_activating_function.batch_norm.beta_arr
            )
            params_list.append(
                self.__graph.hidden_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.__graph.hidden_activating_function.batch_norm.delta_beta_arr
            )
            grads_list.append(
                self.__graph.hidden_activating_function.batch_norm.delta_gamma_arr
            )

        params_list = self.__opt_params.optimize(
            params_list=params_list,
            grads_list=grads_list,
            learning_rate=self.__learning_rate
        )
        self.__graph.visible_bias_arr = params_list.pop(0)
        self.__graph.hidden_bias_arr = params_list.pop(0)
        self.__graph.weights_arr = params_list.pop(0)

        if self.__graph.visible_activating_function.batch_norm is not None:
            self.__graph.visible_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.__graph.visible_activating_function.batch_norm.gamma_arr = params_list.pop(0)

        if self.__graph.hidden_activating_function.batch_norm is not None:
            self.__graph.hidden_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.__graph.hidden_activating_function.batch_norm.gamma_arr = params_list.pop(0)

        self.__graph.visible_diff_bias_arr = np.zeros(self.__graph.visible_bias_arr.shape)
        self.__graph.hidden_diff_bias_arr = np.zeros(self.__graph.hidden_bias_arr.shape)
        self.__graph.diff_weights_arr = np.zeros_like(self.__graph.weights_arr, dtype=np.float64)

    def __h_h_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        Args:
            observed_data_list:      observed data points (feature points).
        '''
        # Waking.
        self.__graph.visible_activity_arr = observed_data_arr.copy()
        
        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            np.dot(
                self.__graph.visible_activity_arr,
                self.__graph.weights_arr
            ) + self.__graph.hidden_bias_arr
        )

        self.__graph.hidden_activity_arr = self.__opt_params.dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr += np.dot(
            self.__graph.visible_activity_arr.T,
            self.__graph.hidden_activity_arr
        )

        self.__graph.visible_diff_bias_arr -= np.nansum(self.__graph.visible_activity_arr, axis=0)
        self.__graph.hidden_diff_bias_arr -= np.nansum(self.__graph.hidden_activity_arr, axis=0)

        # Sleeping.
        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(
            np.dot(
                self.__graph.hidden_activity_arr,
                self.__graph.weights_arr.T
            ) + self.__graph.visible_bias_arr
        )

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            np.dot(
                self.__graph.visible_activity_arr,
                self.__graph.weights_arr
            ) + self.__graph.hidden_bias_arr
        )
        self.__graph.hidden_activity_arr = self.__opt_params.de_dropout(self.__graph.hidden_activity_arr)

        self.__graph.diff_weights_arr -= np.dot(
            self.__graph.visible_activity_arr.T,
            self.__graph.hidden_activity_arr
        )

        self.__graph.visible_diff_bias_arr -= np.nansum(self.__graph.visible_activity_arr, axis=0)
        self.__graph.hidden_diff_bias_arr -= np.nansum(self.__graph.hidden_activity_arr, axis=0)

        self.__graph.diff_weights_arr += self.__opt_params.compute_weight_decay_delta(
            self.__graph.weights_arr
        )

        # Learning.
        params_list= [
            self.__graph.visible_bias_arr,
            self.__graph.hidden_bias_arr,
            self.__graph.weights_arr
        ]
        grads_list = [
            self.__graph.visible_diff_bias_arr,
            self.__graph.hidden_diff_bias_arr,
            self.__graph.diff_weights_arr
        ]

        if self.__graph.visible_activating_function.batch_norm is not None:
            params_list.append(
                self.__graph.visible_activating_function.batch_norm.beta_arr
            )
            params_list.append(
                self.__graph.visible_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.__graph.visible_activating_function.batch_norm.delta_beta_arr
            )
            grads_list.append(
                self.__graph.visible_activating_function.batch_norm.delta_gamma_arr
            )

        if self.__graph.hidden_activating_function.batch_norm is not None:
            params_list.append(
                self.__graph.hidden_activating_function.batch_norm.beta_arr
            )
            params_list.append(
                self.__graph.hidden_activating_function.batch_norm.gamma_arr
            )
            grads_list.append(
                self.__graph.hidden_activating_function.batch_norm.delta_beta_arr
            )
            grads_list.append(
                self.__graph.hidden_activating_function.batch_norm.delta_gamma_arr
            )

        params_list = self.__opt_params.optimize(
            params_list=params_list,
            grads_list=grads_list,
            learning_rate=self.__learning_rate
        )
        self.__graph.visible_bias_arr = params_list.pop(0)
        self.__graph.hidden_bias_arr = params_list.pop(0)
        self.__graph.weights_arr = params_list.pop(0)

        if self.__graph.visible_activating_function.batch_norm is not None:
            self.__graph.visible_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.__graph.visible_activating_function.batch_norm.gamma_arr = params_list.pop(0)

        if self.__graph.hidden_activating_function.batch_norm is not None:
            self.__graph.hidden_activating_function.batch_norm.beta_arr = params_list.pop(0)
            self.__graph.hidden_activating_function.batch_norm.gamma_arr = params_list.pop(0)

        self.__graph.visible_diff_bias_arr = np.zeros(self.__graph.visible_bias_arr.shape)
        self.__graph.hidden_diff_bias_arr = np.zeros(self.__graph.hidden_bias_arr.shape)
        self.__graph.diff_weights_arr = np.zeros_like(self.__graph.weights_arr, dtype=np.float64)

    def __v_h_inference(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Sleeping, waking, and learning.

        Args:
            observed_data_arr:      feature points.
        '''
        self.__graph.hidden_activity_arr = observed_data_arr.copy()

        cdef int split_v_num = int((self.__graph.visible_activity_arr.shape[1] - self.__overlap_n) / 2)
        cdef int split_h_num = int(self.__graph.hidden_activity_arr.shape[1] / 2)

        cdef np.ndarray[DOUBLE_t, ndim=2] left_visible_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] center_visible_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] right_visible_activity_arr

        cdef np.ndarray[DOUBLE_t, ndim=1] center_visible_sum_arr

        if self.__graph.hidden_activity_arr.shape[1] % 2 == 0:
            left_visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr[:, :split_h_num],
                self.__graph.weights_arr.T[:split_h_num, :split_v_num]
            ) + self.__graph.visible_bias_arr[:split_v_num]
            
            center_visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr,
                self.__graph.weights_arr.T
            ) + self.__graph.visible_bias_arr
            
            right_visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr[:, -split_h_num:],
                self.__graph.weights_arr.T[split_h_num:, -split_v_num:]
            ) + self.__graph.visible_bias_arr[-split_v_num:]
        
            # Overlapping for Shape-BM.
            if left_visible_activity_arr.shape[1] + self.__overlap_n + right_visible_activity_arr.shape[1] < self.__graph.visible_activity_arr.shape[1]:
                center_n = self.__overlap_n + 1
            else:
                center_n = self.__overlap_n

            center_visible_sum_arr = np.nansum(center_visible_activity_arr, axis=1)

            self.__graph.visible_activity_arr = np.c_[
                left_visible_activity_arr,
                np.array(
                    [
                        center_visible_sum_arr.reshape(-1, 1)
                    ] * center_n
                ).reshape((center_visible_sum_arr.shape[0], -1)),
                right_visible_activity_arr
            ]
            self.__graph.visible_activity_arr = np.nan_to_num(self.__graph.visible_activity_arr)
        else:
            self.__graph.visible_activity_arr = np.dot(
                self.__graph.hidden_activity_arr,
                self.__graph.weights_arr
            ) + self.__graph.visible_bias_arr

        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(
            self.__graph.visible_activity_arr
        )

        if self.__r_batch_size != -1:
            self.__graph.diff_weights_arr -= np.dot(
                self.__graph.visible_activity_arr.T,
                self.__graph.hidden_activity_arr
            )

            self.__graph.visible_diff_bias_arr -= np.nansum(self.__graph.visible_activity_arr, axis=0)
            self.__graph.hidden_diff_bias_arr -= np.nansum(self.__graph.hidden_activity_arr, axis=0)

            # Waking.
            self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
                np.dot(
                    self.__graph.visible_activity_arr, 
                    self.__graph.weights_arr
                ) + self.__graph.hidden_bias_arr
            )

            self.__graph.diff_weights_arr += np.dot(
                self.__graph.visible_activity_arr.T,
                self.__graph.hidden_activity_arr
            )

            self.__graph.visible_diff_bias_arr += np.nansum(self.__graph.visible_activity_arr, axis=0)
            self.__graph.hidden_diff_bias_arr += np.nansum(self.__graph.hidden_activity_arr, axis=0)

            self.__graph.diff_weights_arr += self.__opt_params.compute_weight_decay_delta(
                self.__graph.weights_arr
            )

            # Learning.
            params_list= [
                self.__graph.visible_bias_arr,
                self.__graph.hidden_bias_arr,
                self.__graph.weights_arr
            ]
            grads_list = [
                self.__graph.visible_diff_bias_arr,
                self.__graph.hidden_diff_bias_arr,
                self.__graph.diff_weights_arr
            ]

            if self.__graph.visible_activating_function.batch_norm is not None:
                params_list.append(
                    self.__graph.visible_activating_function.batch_norm.beta_arr
                )
                params_list.append(
                    self.__graph.visible_activating_function.batch_norm.gamma_arr
                )
                grads_list.append(
                    self.__graph.visible_activating_function.batch_norm.delta_beta_arr
                )
                grads_list.append(
                    self.__graph.visible_activating_function.batch_norm.delta_gamma_arr
                )

            if self.__graph.hidden_activating_function.batch_norm is not None:
                params_list.append(
                    self.__graph.hidden_activating_function.batch_norm.beta_arr
                )
                params_list.append(
                    self.__graph.hidden_activating_function.batch_norm.gamma_arr
                )
                grads_list.append(
                    self.__graph.hidden_activating_function.batch_norm.delta_beta_arr
                )
                grads_list.append(
                    self.__graph.hidden_activating_function.batch_norm.delta_gamma_arr
                )

            params_list = self.__opt_params.optimize(
                params_list=params_list,
                grads_list=grads_list,
                learning_rate=self.__learning_rate
            )
            self.__graph.visible_bias_arr = params_list.pop(0)
            self.__graph.hidden_bias_arr = params_list.pop(0)
            self.__graph.weights_arr = params_list.pop(0)

            if self.__graph.visible_activating_function.batch_norm is not None:
                self.__graph.visible_activating_function.batch_norm.beta_arr = params_list.pop(0)
                self.__graph.visible_activating_function.batch_norm.gamma_arr = params_list.pop(0)

            if self.__graph.hidden_activating_function.batch_norm is not None:
                self.__graph.hidden_activating_function.batch_norm.beta_arr = params_list.pop(0)
                self.__graph.hidden_activating_function.batch_norm.gamma_arr = params_list.pop(0)

            self.__graph.visible_diff_bias_arr = np.zeros(self.__graph.visible_bias_arr.shape)
            self.__graph.hidden_diff_bias_arr = np.zeros(self.__graph.hidden_bias_arr.shape)
            self.__graph.diff_weights_arr = np.zeros_like(self.__graph.weights_arr, dtype=np.float64)

    def __h_h_inference(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Sleeping, waking, and learning.

        Args:
            observed_data_arr:      feature points.
        '''
        # Sleeping.
        self.__graph.hidden_activity_arr = observed_data_arr.copy()
        
        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(
            np.dot(
                self.__graph.hidden_activity_arr,
                self.__graph.weights_arr.T
            ) + self.__graph.hidden_bias_arr
        )

        if self.__r_batch_size != -1:
            self.__graph.diff_weights_arr -= np.dot(
                self.__graph.visible_activity_arr.T,
                self.__graph.hidden_activity_arr
            )

            self.__graph.visible_diff_bias_arr -= np.nansum(self.__graph.visible_activity_arr, axis=0)
            self.__graph.hidden_diff_bias_arr -= np.nansum(self.__graph.hidden_activity_arr, axis=0)

            # Waking.
            self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
                np.dot(
                    self.__graph.visible_activity_arr, 
                    self.__graph.weights_arr
                ) + self.__graph.hidden_bias_arr
            )

            self.__graph.diff_weights_arr += np.dot(
                self.__graph.visible_activity_arr.T,
                self.__graph.hidden_activity_arr
            )

            self.__graph.visible_diff_bias_arr += np.nansum(self.__graph.visible_activity_arr, axis=0)
            self.__graph.hidden_diff_bias_arr += np.nansum(self.__graph.hidden_activity_arr, axis=0)

            self.__graph.diff_weights_arr += self.__opt_params.compute_weight_decay_delta(
                self.__graph.weights_arr
            )

            # Learning.
            params_list= [
                self.__graph.visible_bias_arr,
                self.__graph.hidden_bias_arr,
                self.__graph.weights_arr
            ]
            grads_list = [
                self.__graph.visible_diff_bias_arr,
                self.__graph.hidden_diff_bias_arr,
                self.__graph.diff_weights_arr
            ]

            if self.__graph.visible_activating_function.batch_norm is not None:
                params_list.append(
                    self.__graph.visible_activating_function.batch_norm.beta_arr
                )
                params_list.append(
                    self.__graph.visible_activating_function.batch_norm.gamma_arr
                )
                grads_list.append(
                    self.__graph.visible_activating_function.batch_norm.delta_beta_arr
                )
                grads_list.append(
                    self.__graph.visible_activating_function.batch_norm.delta_gamma_arr
                )

            if self.__graph.hidden_activating_function.batch_norm is not None:
                params_list.append(
                    self.__graph.hidden_activating_function.batch_norm.beta_arr
                )
                params_list.append(
                    self.__graph.hidden_activating_function.batch_norm.gamma_arr
                )
                grads_list.append(
                    self.__graph.hidden_activating_function.batch_norm.delta_beta_arr
                )
                grads_list.append(
                    self.__graph.hidden_activating_function.batch_norm.delta_gamma_arr
                )

            params_list = self.__opt_params.optimize(
                params_list=params_list,
                grads_list=grads_list,
                learning_rate=self.__learning_rate
            )
            self.__graph.visible_bias_arr = params_list.pop(0)
            self.__graph.hidden_bias_arr = params_list.pop(0)
            self.__graph.weights_arr = params_list.pop(0)

            if self.__graph.visible_activating_function.batch_norm is not None:
                self.__graph.visible_activating_function.batch_norm.beta_arr = params_list.pop(0)
                self.__graph.visible_activating_function.batch_norm.gamma_arr = params_list.pop(0)

            if self.__graph.hidden_activating_function.batch_norm is not None:
                self.__graph.hidden_activating_function.batch_norm.beta_arr = params_list.pop(0)
                self.__graph.hidden_activating_function.batch_norm.gamma_arr = params_list.pop(0)

            self.__graph.visible_diff_bias_arr = np.zeros(self.__graph.visible_bias_arr.shape)
            self.__graph.hidden_diff_bias_arr = np.zeros(self.__graph.hidden_bias_arr.shape)
            self.__graph.diff_weights_arr = np.zeros_like(self.__graph.weights_arr, dtype=np.float64)
