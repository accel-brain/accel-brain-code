# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class VisibleLayerInterface(metaclass=ABCMeta):
    '''
    The interface for learning in visible layer.
    '''

    @abstractmethod
    def observe_data_point(self, x):
        '''
        Input obseved data points.

        Args:
            x:  observed data points.
        '''
        raise NotImplementedError()

    @abstractmethod
    def visible_update_state(self, link_value):
        '''
        Update the activity.

        Args:
            link_value:      Input value.

        '''
        raise NotImplementedError()
