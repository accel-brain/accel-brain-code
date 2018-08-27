# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
import pandas as pd

from pycomposer.rbm_annealing import RBMAnnealing

# `Builder` in `Builder Patter`.
from pydbm.dbm.builders.lstm_rt_rbm_simple_builder import LSTMRTRBMSimpleBuilder
# LSTM and Contrastive Divergence for function approximation.
from pydbm.approximation.rtrbmcd.lstm_rt_rbm_cd import LSTMRTRBMCD
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# Stochastic Gradient Descent(SGD) as optimizer.
from pydbm.optimization.optparams.sgd import SGD
# Quantum Monte Carlo(QMC)
from pyqlearning.annealingmodel.quantum_monte_carlo import QuantumMonteCarlo
# Cost function for QMC.
from pyqlearning.annealingmodel.distancecomputable.cost_as_distance import CostAsDistance


class LSTMRTRBMQMC(RBMAnnealing):
    '''
    Composer based on LSTM-RTRBM and Quantum Monte Carlo.
    '''

    def create_rbm(self, visible_num, hidden_num, learning_rate):
        '''
        Build `RestrictedBoltzmmanMachine`.
        
        Args:
            visible_num:    The number of units in visible layer.
            hidden_num:     The number of units in hidden layer.
            learning_rate:  Learning rate.
        
        Returns:
            `RestrictedBoltzmmanMachine`.
        '''
        # `Builder` in `Builder Pattern` for LSTM-RTRBM.
        rnnrbm_builder = LSTMRTRBMSimpleBuilder()
        # Learning rate.
        rnnrbm_builder.learning_rate = learning_rate
        # Set units in visible layer.
        rnnrbm_builder.visible_neuron_part(LogisticFunction(), visible_num)
        # Set units in hidden layer.
        rnnrbm_builder.hidden_neuron_part(LogisticFunction(), hidden_num)
        # Set units in RNN layer.
        rnnrbm_builder.rnn_neuron_part(TanhFunction())
        # Set graph and approximation function, delegating `SGD` which is-a `OptParams`.
        rnnrbm_builder.graph_part(LSTMRTRBMCD(opt_params=SGD()))
        # Building.
        rbm = rnnrbm_builder.get_result()

        return rbm

    def create_annealing(self, cost_functionable, params_arr, cycles_num=100):
        '''
        Build `AnnealingModel`.
        
        Args:
            cost_functionable:      is-a `CostFunctionable`.
            params_arr:             Random sampled parameters.
            cycles_num:             The number of annealing cycles.
        
        Returns:
            `AnnealingModel`.
        
        '''
        # Compute cost as distance for `QuantumMonteCarlo`.
        distance_computable = CostAsDistance(params_arr, cost_functionable)

        # Init.
        annealing_model = QuantumMonteCarlo(
            distance_computable=distance_computable,

            # The number of annealing cycles.
            cycles_num=100,

            # Inverse temperature (Beta).
            inverse_temperature_beta=0.1,

            # Gamma. (so-called annealing coefficient.) 
            gammma=1.0,

            # Attenuation rate for simulated time.
            fractional_reduction=0.99,

            # The dimention of Trotter.
            trotter_dimention=10,

            # The number of Monte Carlo steps.
            mc_step=100,

            # The number of parameters which can be optimized.
            point_num=100,

            # Default `np.ndarray` of 2-D spin glass in Ising model.
            spin_arr=None,

            # Tolerance for the optimization.
            # When the ΔE is not improving by at least `tolerance_diff_e`
            # for two consecutive iterations, annealing will stops.
            tolerance_diff_e=0.01
        )
        
        return annealing_model
