# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np


class AnnealingModel(metaclass=ABCMeta):
    '''
    Abstract class of Annealing.
    '''
    # The fitted data points.
    dist_mat_arr = None

    # The list of Predicted logs.
    predicted_log_list = []

    # Optimized data points.
    x = None

    # The `np.ndarray` of current distribution.
    current_dist_arr = None

    # The `np.ndarray` of current predicted distribution.
    current_cost_arr = None

    # The `np.ndarray` of stocked predicted distribution.
    stocked_predicted_arr = None

    # Accepted pos
    accepted_pos = None

    def fit_dist_mat(self, dist_mat_arr):
        '''
        Fit ovserved data points.

        Args:
            dist_mat_arr:    fitted data points.
        '''
        # Set the data points.
        self.dist_mat_arr = dist_mat_arr

    @abstractmethod
    def annealing(self):
        '''
        Annealing.
        '''
        raise NotImplementedError()
