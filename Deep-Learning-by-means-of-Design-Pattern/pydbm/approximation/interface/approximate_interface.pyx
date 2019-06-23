# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
cimport cython
ctypedef np.float64_t DOUBLE_t


class ApproximateInterface(metaclass=ABCMeta):
    '''
    The interface for function approximations.
    '''

    @abstractproperty
    def reconstruct_error_list(self):
        ''' Reconstruction error. '''
        raise NotImplementedError()

    @abstractmethod
    def approximate_learn(
        self,
        graph,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        double dropout_rate,
        np.ndarray observed_data_arr,
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
            dropout_rate:                   Dropout rate.
            observed_data_arr:              Observed data points.
            training_count:                 Training counts.
            batch_size:                     Batch size (0: not mini-batch)

        Returns:
            Graph of neurons.
        '''
        raise NotImplementedError()

    @abstractmethod
    def approximate_inference(
        self,
        graph,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        double dropout_rate,
        np.ndarray observed_data_arr,
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
            dropout_rate:                   Dropout rate.
            observed_data_arr:              Observed data points.
            training_count:                 Training counts.
            r_batch_size:                   Batch size.
                                            If this value is `0`, the inferencing is a recursive learning.
                                            If this value is more than `0`, the inferencing is a mini-batch recursive learning.
                                            If this value is '-1', the inferencing is not a recursive learning.

        Returns:
            Graph of neurons.
        '''
        raise NotImplementedError()
