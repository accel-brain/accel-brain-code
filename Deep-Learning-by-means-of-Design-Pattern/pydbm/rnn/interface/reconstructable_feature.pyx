# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ReconstructableFeature(metaclass=ABCMeta):
    '''
    The interface for Encoder/Decoder scheme.
    '''

    @abstractmethod
    def inference(
        self,
        np.ndarray[DOUBLE_t, ndim=2] time_series_arr,
        np.ndarray hidden_activity_arr = np.array([]),
        np.ndarray rnn_activity_arr = np.array([])
    ):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            time_series_arr:        Array like or sparse matrix as the observed data ponts.
            hidden_activity_arr:    Array like or sparse matrix as the state in hidden layer.
            rnn_activity_arr:       Array like or sparse matrix as the state in RNN.
        
        Returns:
            Tuple(
                Array like or sparse matrix of reconstructed instances of time-series,
                Array like or sparse matrix of the state in hidden layer,
                Array like or sparse matrix of the state in RNN
            )
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it, 
        considering this method will be called per one cycle in instances of time-series.

        Returns:
            The `list` of array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        raise NotImplementedError()
