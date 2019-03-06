# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class AbstractableSemantics(metaclass=ABCMeta):
    '''
    Automatic abstraction and summarization 
    with the Neural Network language model approach.

    This `interface` is designed the `Strategy Pattern`.

    References:

    '''

    @abstractmethod
    def learn(self, observed_arr, target_arr):
        '''
        Training the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            target_arr:         `np.ndarray` of target labeled data.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def inference(self, observed_arr):
        '''
        Infernece by the model.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of inferenced feature points.
        '''
        raise NotImplementedError("This method must be implemented.")
