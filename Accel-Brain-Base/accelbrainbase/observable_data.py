# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ObservableData(metaclass=ABCMeta):
    '''
    The interface to observe and learn samples, and inference the result.
    '''

    @abstractmethod
    def learn(self, iteratable_data):
        '''
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

        Args:
            iteratable_data:     is-a `IteratableData`.
        '''
        raise NotImplementedError()

    @abstractmethod
    def inference(self, observed_arr):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:     Observed data points.
        
        Returns:
            Inferenced results.
        '''
        raise NotImplementedError()
