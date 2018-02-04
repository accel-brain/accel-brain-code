# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class HiddenLayerInterface(metaclass=ABCMeta):
    '''
    The interface for executing learning in hidden layer.
    '''

    @abstractmethod
    def hidden_update_state(self, double link_value):
        '''
        Update activity.

        Args:
            link_value:      Input value

        '''
        raise NotImplementedError()
