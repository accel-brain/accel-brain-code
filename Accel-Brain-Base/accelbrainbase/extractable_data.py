# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ExtractableData(metaclass=ABCMeta):
    '''
    The interface to extract samples from dataset.
    '''

    @abstractmethod
    def extract(
        self,
        path,
    ):
        '''
        Extract data.

        Args:
            path:       `str` of source path.
        
        Returns:
            Tensor.
        '''
        raise NotImplementedError()
