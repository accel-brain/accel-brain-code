# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ApproximateInterface(metaclass=ABCMeta):
    '''
    The interface for function approximations.
    '''

    @abstractmethod
    def approximate_learn(
        self,
        graph,
        learning_rate,
        observed_data_arr,
        traning_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            observed_data_arr:    observed data points.
            traning_count:        Training counts.

        Returns:
            Graph of neurons.
        '''
        raise NotImplementedError()
