# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class InferableMelody(metaclass=ABCMeta):
    '''
    The interface for inferacing next melody.
    '''
    
    @abstractmethod
    def inferance(self, midi_df, octave=6):
        '''
        Inferance next melody.
        
        Args:
            midi_df:    `pd.DataFrame` of MIDI file.
        
        Returns:
            `pd.DataFrame` of MIDI file.
        '''
        raise NotImplementedError("This method must be implemented.")
