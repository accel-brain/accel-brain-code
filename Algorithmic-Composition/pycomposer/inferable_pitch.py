# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class InferablePitch(metaclass=ABCMeta):
    '''
    The interface for inferacing next pitch.
    '''
    
    @abstractmethod
    def inferance(self, pitch_arr):
        '''
        Inferance and select next pitch of `pre_pitch` from the values of `pitch_arr`.
        
        Args:
            pitch_arr:    `np.ndarray` of pitch.
        
        Returns:
            `np.ndarray` of pitch.
        '''
        raise NotImplementedError("This method must be implemented.")
