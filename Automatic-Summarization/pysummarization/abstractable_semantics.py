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
    def learn(self, iteratable_data):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Args:
            iteratable_data:     is-a `IteratableData`.

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

    @abstractmethod
    def summarize(self, test_arr, vectorizable_token, sentence_list, limit=5):
        '''
        Summarize input document.

        Args:
            test_arr:               `np.ndarray` of observed data points..
            vectorizable_token:     is-a `VectorizableToken`.
            sentence_list:          `list` of all sentences.
            limit:                  The number of selected abstract sentence.
        
        Returns:
            `np.ndarray` of scores.
        '''
        raise NotImplementedError("This method must be implemented.")
