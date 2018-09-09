# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ExtractableMelody(metaclass=ABCMeta):
    '''
    The interface melody extractor.
    '''
    
    @abstractmethod
    def extract(self, midi_arr):
        '''
        
        '''
        raise NotImplementedError()
