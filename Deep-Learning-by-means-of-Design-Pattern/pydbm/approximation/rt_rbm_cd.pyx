# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from logging import getLogger
import warnings
cimport cython
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
from pydbm.optimization.opt_params import OptParams
from pydbm.optimization.optparams.sgd import SGD
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.loss.mean_squared_error import MeanSquaredError
ctypedef np.float64_t DOUBLE_t


class RTRBMCD(ApproximateInterface):
    '''
    Recurrent Temporal Restricted Boltzmann Machines
    based on Contrastive Divergence.

    Conceptually, the positive phase is to the negative phase what waking is to sleeping.

    The RTRBM (Sutskever, I., et al. 2009) is a probabilistic 
    time-series model which can be viewed as a temporal stack of RBMs, 
    where each RBM has a contextual hidden state that is received 
    from the previous RBM and is used to modulate its hidden units bias.

    Parameters:
        graph.weights_arr:                $W$ (Connection between v^{(t)} and h^{(t)})
        graph.visible_bias_arr:           $b_v$ (Bias in visible layer)
        graph.hidden_bias_arr:            $b_h$ (Bias in hidden layer)
        graph.rnn_hidden_weights_arr:     $W'$ (Connection between h^{(t-1)} and b_h^{(t)})
        graph.rnn_visible_weights_arr:    $W''$ (Connection between h^{(t-1)} and b_v^{(t)})
        graph.hat_hidden_activity_arr:    $\hat{h}^{(t)}$ (RNN with hidden units)
        graph.pre_hidden_activity_arr:    $\hat{h}^{(t-1)}$
    
    References:
        - Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. arXiv preprint arXiv:1206.6392.
        - Lyu, Q., Wu, Z., Zhu, J., & Meng, H. (2015, June). Modelling High-Dimensional Sequences with LSTM-RTRBM: Application to Polyphonic Music Generation. In IJCAI (pp. 4138-4139).
        - Lyu, Q., Wu, Z., & Zhu, J. (2015, October). Polyphonic music modelling with LSTM-RTRBM. In Proceedings of the 23rd ACM international conference on Multimedia (pp. 991-994). ACM.
        - Sutskever, I., Hinton, G. E., & Taylor, G. W. (2009). The recurrent temporal restricted boltzmann machine. In Advances in Neural Information Processing Systems (pp. 1601-1608).

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
    graph = None
    # Learning rate.
    learning_rate = 0.5
    # Batch size in learning.
    batch_size = 0
    # Batch step in learning.
    batch_step = 0
    # Batch size in inference(recursive learning or not).
    r_batch_size = 0
    # Batch step in inference(recursive learning or not).
    r_batch_step = 0
    # visible activity in negative phase.
    negative_visible_activity_arr = None

    def __init__(self, opt_params=None, computable_loss=None):
        '''
        Init.
        
        Args:
            opt_params:         is-a `OptParams`.
            computable_loss:    is-a `ComputableLoss`.

        '''
        if opt_params is None:
            opt_params = SGD(momentum=0.0)

        if computable_loss is None:
            computable_loss = MeanSquaredError()

        if isinstance(opt_params, OptParams):
            self.__opt_params = opt_params
        else:
            raise TypeError()

        if isinstance(computable_loss, ComputableLoss):
            self.__computable_loss = computable_loss
        else:
            raise TypeError()

        logger = getLogger("pydbm")
        self.__logger = logger

    def approximate_learn(
        self,
        graph,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        np.ndarray[DOUBLE_t, ndim=3] observed_data_arr,
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
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] inferenced_arr = np.empty((
            batch_size,
            observed_data_arr.shape[1],
            observed_data_arr.shape[2]
        ))
        cdef int batch_index

        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.graph = graph
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        reconstruct_error_list = []

        # Learning.
        for epoch in range(training_count):
            if ((epoch + 1) % attenuate_epoch == 0):
                self.learning_rate = self.learning_rate * learning_attenuate_rate

            rand_index = np.random.choice(observed_data_arr.shape[0], size=batch_size)
            batch_observed_arr = observed_data_arr[rand_index]
            for cycle_index in range(batch_observed_arr.shape[1]):
                # RNN learning.
                self.rnn_learn(batch_observed_arr[:, cycle_index])
                # Wake and sleep.
                self.wake_sleep_learn(self.graph.visible_activity_arr)
                # Memorizing.
                self.memorize_activity(
                    batch_observed_arr[:, cycle_index],
                    self.graph.visible_activity_arr
                )
                inferenced_arr[:, cycle_index] = self.graph.visible_activity_arr

            reconstruct_error = self.compute_loss(batch_observed_arr, inferenced_arr)
            self.__logger.debug("Epoch: " + str(epoch) + " Reconstruction Error: " + str(reconstruct_error))
            reconstruct_error_list.append(reconstruct_error)
            # Back propagation.
            self.back_propagation()

        self.graph.reconstruct_error_arr = np.array(reconstruct_error_list)
        return self.graph

    def approximate_inference(
        self,
        graph,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        np.ndarray[DOUBLE_t, ndim=3] observed_data_arr,
        int traning_count=-1,
        int r_batch_size=200,
        int training_count=1000,
        seq_len=None
    ):
        '''
        Inference with function approximation.

        Args:
            graph:                          Graph of neurons.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            observed_data_arr:              observed data points.
            training_count:                 Training counts.
            r_batch_size:                   Batch size.
                                            If this value is `0`, the inferencing is a recursive learning.
                                            If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                            If this value is '-1', the inferencing is not a recursive learning.

            seq_len:                        The length of sequences.
                                            If `None`, this value will be considered as `observed_data_arr.shape[1]`.

        Returns:
            Graph of neurons.
        '''
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef int batch_index
        cdef int batch_size

        if r_batch_size <= 0:
            batch_size = observed_data_arr.shape[0]
        else:
            batch_size = r_batch_size

        if seq_len is None:
            seq_len = observed_data_arr.shape[1]

        cdef np.ndarray[DOUBLE_t, ndim=3] inferenced_arr = np.empty((
            batch_size,
            seq_len,
            observed_data_arr.shape[2]
        ))
        cdef np.ndarray[DOUBLE_t, ndim=3] feature_points_arr = np.empty((
            batch_size,
            seq_len,
            self.graph.hidden_activity_arr.shape[1]
        ))

        if traning_count != -1:
            training_count = traning_count
            warnings.warn("`traning_count` will be removed in future version. Use `training_count`.", FutureWarning)

        self.graph = graph
        self.learning_rate = learning_rate
        self.r_batch_size = r_batch_size

        for epoch in range(training_count):
            if ((epoch + 1) % attenuate_epoch == 0):
                self.learning_rate = self.learning_rate * learning_attenuate_rate

            if r_batch_size > 0:
                rand_index = np.random.choice(observed_data_arr.shape[0], size=r_batch_size)
                batch_observed_arr = observed_data_arr[rand_index]
            else:
                batch_observed_arr = observed_data_arr

            for cycle_index in range(seq_len):
                # RNN learning.
                self.rnn_learn(batch_observed_arr[:, cycle_index])
                self.memorize_activity(
                    batch_observed_arr[:, cycle_index],
                    self.graph.visible_activity_arr
                )
                self.wake_sleep_inference(self.graph.visible_activity_arr)

                inferenced_arr[:, cycle_index] = self.graph.visible_activity_arr
                feature_points_arr[:, cycle_index] = self.graph.hidden_activity_arr

            # Back propagation.
            if r_batch_size >= 0:
                self.back_propagation()

        self.graph.reconstructed_arr = inferenced_arr
        self.graph.inferenced_arr = inferenced_arr[:, -1]
        self.graph.feature_points_arr = feature_points_arr
        return self.graph

    def rnn_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Learning for RNN.

        Args:
            observed_data_list:      observed data points.
        '''
        self.graph.visible_activity_arr = observed_data_arr.copy()

        if self.graph.pre_hidden_activity_arr.shape[0] == 0:
            return
        if self.graph.hat_hidden_activity_arr.shape[0] == 0:
            return

        self.graph.rnn_hidden_bias_arr = np.dot(
            self.graph.hat_hidden_activity_arr,
            self.graph.rnn_hidden_weights_arr
        ) + self.graph.hidden_bias_arr
        
        self.graph.rnn_visible_bias_arr = np.dot(
            self.graph.hat_hidden_activity_arr,
            self.graph.rnn_visible_weights_arr.T
        ) + self.graph.visible_bias_arr
        
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.pre_hidden_activity_arr,
                self.graph.rnn_hidden_weights_arr
            ) + self.graph.hidden_bias_arr
        )
        
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            np.dot(
                self.graph.pre_hidden_activity_arr,
                self.graph.rnn_visible_weights_arr.T
            ) + self.graph.visible_bias_arr
        )

    def memorize_activity(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_data_arr,
        np.ndarray[DOUBLE_t, ndim=2] negative_visible_activity_arr
    ):
        '''
        Memorize activity.

        Args:
            observed_data_arr:                Observed data points in positive phase.
            negative_visible_activity_arr:    visible acitivty in negative phase.
        '''
        self.graph.pre_hidden_activity_arr = self.graph.hat_hidden_activity_arr
        
        self.graph.hat_hidden_activity_arr = self.graph.rnn_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr.T
        )

        self.graph.visible_diff_bias_arr += np.nansum((observed_data_arr - negative_visible_activity_arr), axis=0)

    def compute_loss(self, np.ndarray batch_observed_arr, np.ndarray inferenced_arr):
        '''
        Compute loss.

        Args:
            batch_observed_arr:     `np.ndarray` of observed data points.
            inferenced_arr:         `np.ndarray` of reconstructed feature points.
        
        Returns:
            loss.
        '''
        reconstruct_error = self.__computable_loss.compute_loss(batch_observed_arr, inferenced_arr)
        reconstruct_error += self.__opt_params.compute_weight_decay(
            self.graph.weights_arr
        )
        reconstruct_error += self.__opt_params.compute_weight_decay(
            self.graph.rnn_hidden_weights_arr
        )
        reconstruct_error += self.__opt_params.compute_weight_decay(
            self.graph.rnn_visible_weights_arr
        )
        return reconstruct_error

    def back_propagation(self):
        '''
        Details of the backpropagation through time algorithm.
        '''
        # Learning.
        cdef np.ndarray[DOUBLE_t, ndim=2] visible_step_arr = (
            self.graph.visible_activity_arr + self.graph.visible_diff_bias_arr
        )
        
        cdef np.ndarray[DOUBLE_t, ndim=2] visible_step_activity_arr = self.graph.rnn_activating_function.activate(
            np.dot(
                visible_step_arr,
                self.graph.weights_arr
            ) - self.graph.hidden_bias_arr
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] visible_negative_arr = self.graph.rnn_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr.T
        )
        self.graph.hidden_diff_bias_arr += (
            np.nansum(visible_step_activity_arr, axis=0) - np.nansum(visible_negative_arr, axis=0)
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] step_arr = np.dot(
            visible_step_activity_arr.T,
            visible_step_arr
        )
        
        cdef np.ndarray[DOUBLE_t, ndim=2] negative_arr = np.dot(
            visible_negative_arr.T,
            self.graph.visible_activity_arr
        )
        
        self.graph.diff_weights_arr += (step_arr - negative_arr).T

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_rnn_hidden_weight_arr = np.dot(
            (visible_step_activity_arr - visible_negative_arr).T,
            self.graph.pre_hidden_activity_arr
        )

        delta_rnn_visible_weight_arr = np.dot(
            self.graph.visible_diff_bias_arr.reshape(-1, 1),
            np.nansum(self.graph.pre_hidden_activity_arr, axis=0).reshape(-1, 1).T
        )

        self.graph.diff_weights_arr += self.__opt_params.compute_weight_decay_delta(
            self.graph.weights_arr
        )
        delta_rnn_hidden_weight_arr += self.__opt_params.compute_weight_decay_delta(
            self.graph.rnn_hidden_weights_arr
        )
        delta_rnn_visible_weight_arr += self.__opt_params.compute_weight_decay_delta(
            self.graph.rnn_visible_weights_arr
        )

        params_list = self.__opt_params.optimize(
            params_list=[
                self.graph.visible_bias_arr,
                self.graph.hidden_bias_arr,
                self.graph.weights_arr,
                self.graph.rnn_hidden_weights_arr,
                self.graph.rnn_visible_weights_arr
            ],
            grads_list=[
                self.graph.visible_diff_bias_arr,
                self.graph.hidden_diff_bias_arr,
                self.graph.diff_weights_arr,
                delta_rnn_hidden_weight_arr,
                delta_rnn_visible_weight_arr
            ],
            learning_rate=self.learning_rate
        )
        self.graph.visible_bias_arr = params_list[0]
        self.graph.hidden_bias_arr = params_list[1]
        self.graph.weights_arr = params_list[2]
        self.graph.rnn_hidden_weights_arr = params_list[3]
        self.graph.rnn_visible_weights_arr = params_list[4]

        self.graph.visible_diff_bias_arr = np.zeros(self.graph.visible_bias_arr.shape)
        self.graph.hidden_diff_bias_arr = np.zeros(self.graph.hidden_bias_arr.shape)
        self.graph.diff_weights_arr = np.zeros_like(self.graph.weights_arr, dtype=np.float64)

    def wake_sleep_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        The binary activity is unsupported.

        Args:
            observed_data_list:      observed data points.
        '''
        self.graph.visible_activity_arr = observed_data_arr

        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr
        )

        self.graph.hidden_activity_arr = self.__opt_params.dropout(self.graph.hidden_activity_arr)

        self.graph.diff_weights_arr += np.dot(
            self.graph.visible_activity_arr.T,
            self.graph.hidden_activity_arr
        )

        self.graph.visible_diff_bias_arr += np.nansum(self.graph.visible_activity_arr, axis=0)
        self.graph.hidden_diff_bias_arr += np.nansum(self.graph.hidden_activity_arr, axis=0)

        # Sleeping.
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            np.dot(
                self.graph.hidden_activity_arr,
                self.graph.weights_arr.T
            ) + self.graph.visible_bias_arr
        )

        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr
        )

        self.graph.hidden_activity_arr = self.__opt_params.de_dropout(self.graph.hidden_activity_arr)

        self.graph.diff_weights_arr -= np.dot(
            self.graph.visible_activity_arr.T,
            self.graph.hidden_activity_arr
        )

        self.graph.visible_diff_bias_arr -= np.nansum(self.graph.visible_activity_arr, axis=0)
        self.graph.hidden_diff_bias_arr -= np.nansum(self.graph.hidden_activity_arr, axis=0)

    def wake_sleep_inference(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Sleeping, waking, and inferencing.

        Args:
            observed_data_arr:      feature points.
        '''
        self.graph.visible_activity_arr = observed_data_arr
        
        self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
            np.dot(
                self.graph.visible_activity_arr,
                self.graph.weights_arr
            ) + self.graph.hidden_bias_arr
        )

        if self.r_batch_size != -1:
            self.graph.hidden_activity_arr = self.__opt_params.dropout(self.graph.hidden_activity_arr)
            self.graph.diff_weights_arr += np.dot(
                self.graph.visible_activity_arr.T,
                self.graph.hidden_activity_arr
            )
            self.graph.visible_diff_bias_arr += np.nansum(self.graph.visible_activity_arr, axis=0)
            self.graph.hidden_diff_bias_arr += np.nansum(self.graph.hidden_activity_arr, axis=0)

        # Sleeping.
        self.graph.visible_activity_arr = self.graph.visible_activating_function.activate(
            np.dot(
                self.graph.hidden_activity_arr,
                self.graph.weights_arr.T
            ) + self.graph.visible_bias_arr
        )

        if self.r_batch_size != -1:
            self.graph.hidden_activity_arr = self.graph.hidden_activating_function.activate(
                np.dot(
                    self.graph.visible_activity_arr,
                    self.graph.weights_arr
                ) + self.graph.hidden_bias_arr
            )
            self.graph.hidden_activity_arr = self.__opt_params.de_dropout(self.graph.hidden_activity_arr)

            self.graph.diff_weights_arr -= np.dot(
                self.graph.visible_activity_arr.T, 
                self.graph.hidden_activity_arr
            )
            self.graph.visible_diff_bias_arr -= np.nansum(self.graph.visible_activity_arr, axis=0)
            self.graph.hidden_diff_bias_arr -= np.nansum(self.graph.hidden_activity_arr, axis=0)

    def get_computable_loss(self):
        ''' getter '''
        return self.__computable_loss
    
    def set_computable_loss(self, value):
        ''' setter '''
        self.__computable_loss = value

    computable_loss = property(get_computable_loss, set_computable_loss)

    def get_opt_params(self):
        ''' getter '''
        return self.__opt_params

    def set_opt_params(self, value):
        ''' setter '''
        self.__opt_params = value

    opt_params = property(get_opt_params, set_opt_params)
