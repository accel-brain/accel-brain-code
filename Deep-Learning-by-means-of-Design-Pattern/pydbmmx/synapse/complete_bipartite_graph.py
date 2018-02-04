# -*- coding: utf-8 -*-
from pydbm.synapse_list import Synapse


class CompleteBipartiteGraph(Synapse):
    '''
    Complete Bipartite Graph.
    
    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.
    '''
    def get_visible_activity_arr(self):
        ''' getter '''
        return self.shallower_activity_arr

    def set_visible_activity_arr(self, value):
        ''' setter '''
        self.shallower_activity_arr = value

    visible_activity_arr = property(get_visible_activity_arr, set_visible_activity_arr)

    def get_hidden_activity_arr(self):
        ''' getter '''
        return self.deeper_activity_arr

    def set_hidden_activity_arr(self, value):
        ''' setter '''
        self.deeper_activity_arr = value

    hidden_activity_arr = property(get_hidden_activity_arr, set_hidden_activity_arr)

    def update(self, learning_rate):
        '''
        Update weights.

        Args:
            learning_rate:  Learning rate.
        '''
        if isinstance(learning_rate, float) is False and isinstance(learning_rate, int) is False:
            raise TypeError()

        if self.diff_weights_arr is None:
            self.diff_weights_arr = self.visible_activity_arr * self.hidden_activity_arr.T * learning_rate
        else:
            self.diff_weights_arr += self.visible_activity_arr * self.hidden_activity_arr.T * learning_rate
