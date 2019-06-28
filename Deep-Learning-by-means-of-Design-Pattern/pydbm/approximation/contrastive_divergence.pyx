# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
cimport cython
from pydbm.approximation.interface.approximate_interface import ApproximateInterface
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.optimization.opt_params import OptParams
from pydbm.optimization.optparams.sgd import SGD
ctypedef np.float64_t DOUBLE_t


class ContrastiveDivergence(ApproximateInterface):
    '''
    Contrastive Divergence.
    
    Conceptually, the positive phase is to the negative phase what waking is to sleeping.
    
    In relation to RBM, Contrastive Divergence(CD) is a method for approximation of 
    the gradients of the log-likelihood(Hinton, G. E. 2002).
    
    The procedure of this method is similar to Markov Chain Monte Carlo method(MCMC).
    However, unlike MCMC, the visbile variables to be set first in visible layer is 
    not randomly initialized but the observed data points in training dataset are set 
    to the first visbile variables. And, like Gibbs sampler, drawing samples from hidden 
    variables and visible variables is repeated k times. Empirically (and surprisingly), 
    `k` is considered to be `1`.
    
    References:
        - Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.
    '''

    # The list of the reconstruction error rate.
    __reconstruct_error_list = []
    # Graph of neurons.
    __graph = None
    # Learning rate.
    __learning_rate = 0.5
    # Dropout rate.
    __dropout_rate = 0.5
    # Batch size in learning.
    __batch_size = 0
    # Batch size in inference(recursive learning or not).
    __r_batch_size = 0

    def __init__(
        self,
        computable_loss=None,
        opt_params=None
    ):
        '''
        Init.
        
        Args:
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
        '''
        if computable_loss is None:
            computable_loss = MeanSquaredError()
        if opt_params is None:
            opt_params = SGD(momentum=0.0)
            
        if isinstance(computable_loss, ComputableLoss):
            self.__computable_loss = computable_loss
        else:
            raise TypeError()

        if isinstance(opt_params, OptParams):
            self.__opt_params = opt_params
        else:
            raise TypeError()

        self.__reconstruct_error_list = []

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
            observed_data_arr:              observed data points.
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
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__batch_size = batch_size

        cdef int epoch
        for epoch in range(training_count):
            if ((epoch + 1) % attenuate_epoch == 0):
                self.__learning_rate = self.__learning_rate * learning_attenuate_rate

            if self.__batch_size > 0:
                rand_index = np.random.choice(observed_data_arr.shape[0], size=self.__batch_size)
                self.__wake_sleep_learn(observed_data_arr[rand_index])
            else:
                self.__wake_sleep_learn(observed_data_arr)

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
            observed_data_arr:              observed data points.
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
                self.__sleep_wake_learn(observed_data_arr[rand_index])
            else:
                self.__sleep_wake_learn(observed_data_arr)

        return self.__graph

    def __wake_sleep_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
        '''
        Waking, sleeping, and learning.

        Standing on the premise that the settings of
        the activation function and weights operation are common.

        Args:
            observed_data_arr:      observed data points.
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

        self.__graph.visible_diff_bias_arr += np.nansum(self.__graph.visible_activity_arr, axis=0)
        self.__graph.hidden_diff_bias_arr += np.nansum(self.__graph.hidden_activity_arr, axis=0)

        # Sleeping.
        self.__graph.visible_activity_arr = self.__graph.visible_activating_function.activate(
            np.dot(
                self.__graph.hidden_activity_arr,
                self.__graph.weights_arr.T
            ) + self.__graph.visible_bias_arr
        )

        # Validation.
        loss = self.__computable_loss.compute_loss(observed_data_arr, self.__graph.visible_activity_arr).mean()
        loss += self.__opt_params.compute_weight_decay(
            self.__graph.weights_arr
        )
        self.__reconstruct_error_list.append(loss)

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

    def __sleep_wake_learn(self, np.ndarray[DOUBLE_t, ndim=2] observed_data_arr):
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
            ) + self.__graph.visible_bias_arr
        )

        self.__graph.hidden_activity_arr = self.__graph.hidden_activating_function.activate(
            np.dot(
                self.__graph.visible_activity_arr, 
                self.__graph.weights_arr
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
            params_list = [
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

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This is read-only.")

    def get_reconstruct_error_list(self):
        ''' getter '''
        return self.__reconstruct_error_list

    reconstruct_error_list = property(get_reconstruct_error_list, set_readonly)
