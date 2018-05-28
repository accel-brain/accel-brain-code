# -*- coding: utf-8 -*-
from pyqlearning.annealingmodel.cost_functionable import CostFunctionable
from pyqlearning.qlearning.boltzmann_q_learning import BoltzmannQLearning
from copy import copy
import pandas as pd


class GreedyQLearningCost(CostFunctionable):
    '''
    Cost function for Boltzmann Q-Learning
    which is-a `CostFunctionable` in relation to `AnnealingModel`.
    '''
    
    __init_state_key = None
    
    def __init__(
        self,
        greedy_q_learning,
        init_state_key
    ):
        '''
        Init.
        
        Args:
            boltzmann_q_learning:  is-a `BoltzmannQLearning`.
            init_state_key:        First state key.
        '''
        if isinstance(boltzmann_q_learning, BoltzmannQLearning):
            self.__boltzmann_q_learning = boltzmann_q_learning
        else:
            raise TypeError()

        self.__init_state_key = init_state_key
    
    def compute(self, x):
        '''
        Compute cost.
        
        Args:
            x:    `np.ndarray` of explanatory variables.
        
        Returns:
            cost
        '''
        q_learning = copy(self.__greedy_q_learning)
        q_learning.alpha_value = x[0]
        q_learning.gamma_value = x[1]
        if self.__init_state_key is not None:
            q_learning.learn(state_key=self.__init_state_key, limit=int(x[2]))
        else:
            q_learning.learn(limit=x[2])
        q_sum = q_learning.q_df.q_value.sum()
        if q_sum != 0:
            cost = q_learning.q_df.shape[0] / q_sum
        else:
            cost = q_learning.q_df.shape[0] / 1e-4

        return cost
