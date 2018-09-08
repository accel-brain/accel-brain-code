# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ClusterableDoc(metaclass=ABCMeta):
    '''
    Document clustering.
    '''
    
    @abstractmethod
    def learn(self, document_arr):
        '''
        Learning.

        Args:
            document_arr:    `np.ndarray` of sentences vectors.

        Retruns:
            `np.ndarray` of labeled data.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def inference(self, document_arr):
        '''
        Inferencing.

        Args:
            document_arr:    `np.ndarray` of sentences vectors.

        Retruns:
            `np.ndarray` of labeled data.
        '''
        raise NotImplementedError("This method must be implemented.")
