# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class InferableDuration(metaclass=ABCMeta):
    '''
    The interface for inferacing the duration.
    '''
    
    @abstractmethod
    def inference(
        self,
        chord_time,
        measure,
        metronome_time,
        measure_n,
        beat_n,
        total_measure_n
    ):
        '''
        Inference the duration.
        
        Args:
            df:    `pd.DataFrame`.
        
        Returns:
            .
        '''
        raise NotImplementedError("This method must be implemented.")
