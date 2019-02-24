# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
ctypedef np.float64_t DOUBLE_t


class ReconstructableModel(metaclass=ABCMeta):
    '''
    The interface of reconstructable model.
    '''

    @abstractmethod
    def learn(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr, np.ndarray target_arr=np.array([])):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Override.

        Args:
            observed_arr:    Array like or sparse matrix as the observed data points.
            target_arr:      Array like or sparse matrix as the target data points.
                             To learn as Auto-encoder, this value must be `None` or equivalent to `observed_arr`.
        '''
        raise NotImplementedError()

    @abstractmethod
    def inference(
        self,
        np.ndarray[DOUBLE_t, ndim=3] observed_arr,
        np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr=None,
        np.ndarray[DOUBLE_t, ndim=2] cec_activity_arr=None
    ):
        '''
        Inference the feature points to reconstruct the time-series.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.
            hidden_activity_arr:    Array like or sparse matrix as the state in hidden layer.
            cec_activity_arr:       Array like or sparse matrix as the state in RNN.

        Returns:
            Tuple data.
            - Array like or sparse matrix of reconstructed instances of time-series,
            - Array like or sparse matrix of the state in hidden layer,
            - Array like or sparse matrix of the state in RNN.
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_feature_points(self):
        '''
        Extract feature points.
        
        Returns:
            Array like or sparse matrix of feature points.
        '''
        raise NotImplementedError()

    @abstractmethod
    def hidden_back_propagate(self, np.ndarray[DOUBLE_t, ndim=2] delta_output_arr):
        '''
        Back propagation in hidden layer.
        
        Args:
            delta_output_arr:    Delta.
        
        Returns:
            Tuple data.
            - `np.ndarray` of Delta, 
            - `list` of gradations.
        '''
        raise NotImplementedError()

    @abstractmethod
    def save_pre_learned_params(self, dir_name, file_name=None):
        '''
        Save pre-learned parameters.
        
        Args:
            dir_name:   Path of dir. If `None`, the file is saved in the current directory.
            file_name:  File name.
        '''
        raise NotImplementedError()

    @abstractmethod
    def load_pre_learned_params(self, dir_name, file_name=None):
        '''
        Load pre-learned parameters.
        
        Args:
            dir_name:   Path of dir. If `None`, the file is saved in the current directory.
            file_name:  File name.
        '''
        raise NotImplementedError()

    @abstractproperty
    def opt_params(self):
        ''' is-a OptParams '''
        raise NotImplementedError()
