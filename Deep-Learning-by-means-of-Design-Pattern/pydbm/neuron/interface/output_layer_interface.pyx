# -*- coding: utf-8 -*-
import pyximport; pyximport.install()
from abc import ABCMeta, abstractmethod


class OutputLayerInterface(metaclass=ABCMeta):
    '''
    The interface for learning in output layer.
    '''

    @abstractmethod
    def output_update_state(self, double link_value):
        '''
        Update activity.

        Args:
            link_value:      Input value.

        '''
        raise NotImplementedError()

    @abstractmethod
    def release(self):
        '''
        Release the activity.

        Returns:
            The activity.
        '''
        raise NotImplementedError()
