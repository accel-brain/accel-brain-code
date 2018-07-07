# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class ReconstructableFeature(metaclass=ABCMeta):
    '''
    The interface for Encoder/Decoder scheme.
    '''

    @abstractmethod
    def inference(self, time_series_X_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            time_series_X_arr:    Array like or sparse matrix as the observed data ponts.
        
        Returns:
            Array like or sparse matrix of reconstructed instances of time-series.
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
