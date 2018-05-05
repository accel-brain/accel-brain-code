# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class InferablePitch(metaclass=ABCMeta):
    '''
    The interface for inferacing next pitch.
    '''
    
    @abstractmethod
    def inferance(self, pre_pitch, pitch_arr):
        '''
        Inferance and select next pitch of `pre_pitch` from the values of `pitch_arr`.
        
        Args:
            pre_pitch:    The pitch in `t-1`.
            pitch_arr:    The list of selected pitchs.
        
        Returns:
            The pitch in `t`.
        '''
        raise NotImplementedError("This method must be implemented.")
