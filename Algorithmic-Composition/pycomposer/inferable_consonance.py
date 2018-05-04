# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class InferableConsonance(metaclass=ABCMeta):
    '''
    The interface for inferacing the Consonances.
    '''
    
    @abstractmethod
    def inference(self, pre_pitch, pitch):
        '''
        Inference the degree of Consonance.
        
        Args:
            pre_pitch:    The pitch in `t-1`.
            limit:        The number of return list.
        
        Returns:
            The list of pitchs in `t`.
        '''
        raise NotImplementedError("This method must be implemented.")
