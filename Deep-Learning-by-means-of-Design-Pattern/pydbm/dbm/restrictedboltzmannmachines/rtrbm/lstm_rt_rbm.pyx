# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
import warnings
from pydbm.dbm.restrictedboltzmannmachines.rt_rbm import RTRBM
ctypedef np.float64_t DOUBLE_t


class LSTMRTRBM(RTRBM):
    '''
    LSTM-RTRBM.
    '''

    def extract_transfered_params(self, double loc=0.0, double scale=1.0):
        '''
        Extract transfered parameters.
        
        Args:
            loc:    Mean (“centre”) of the distribution.
            scale:  Standard deviation (spread or “width”) of the distribution.
        
        Returns:
            `Synapse` which has pre-learned parameters.
        '''
        self.graph.weights_xg_arr = self.graph.v_hat_weights_arr + np.random.normal(
            size=self.graph.v_hat_weights_arr.shape, loc=loc, scale=scale
        )
        self.graph.weights_xi_arr = self.graph.v_hat_weights_arr + np.random.normal(
            size=self.graph.v_hat_weights_arr.shape, loc=loc, scale=scale
        )
        self.graph.weights_xf_arr = self.graph.v_hat_weights_arr + np.random.normal(
            size=self.graph.v_hat_weights_arr.shape, loc=loc, scale=scale
        )
        self.graph.weights_xo_arr = self.graph.v_hat_weights_arr + np.random.normal(
            size=self.graph.v_hat_weights_arr.shape, loc=loc, scale=scale
        )
        
        self.graph.weights_hg_arr = self.graph.rbm_hidden_weights_arr + np.random.normal(
            size=self.graph.rbm_hidden_weights_arr.shape, loc=loc, scale=scale
        )
        self.graph.weights_hi_arr = self.graph.rbm_hidden_weights_arr + np.random.normal(
            size=self.graph.rbm_hidden_weights_arr.shape, loc=loc, scale=scale
        )
        self.graph.weights_hf_arr = self.graph.rbm_hidden_weights_arr + np.random.normal(
            size=self.graph.rbm_hidden_weights_arr.shape, loc=loc, scale=scale
        )
        self.graph.weights_ho_arr = self.graph.rbm_hidden_weights_arr + np.random.normal(
            size=self.graph.rbm_hidden_weights_arr.shape, loc=loc, scale=scale
        )
        self.graph.weights_hy_arr = self.graph.rbm_hidden_weights_arr + np.random.normal(
            size=self.graph.rbm_hidden_weights_arr.shape, loc=loc, scale=scale
        )

        return self.graph
