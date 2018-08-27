# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pyqlearning.annealingmodel.cost_functionable import CostFunctionable


class RBMCostFunction(CostFunctionable):
    '''
    Cost function for RBM.
    '''
    
    __epoch = 0
    
    def __init__(self, inferenced_arr, rbm, batch_size):
        
        self.__inferenced_arr = inferenced_arr
        self.__rbm = rbm
        self.__batch_size = batch_size
        
        self.__epoch = 0
        logger = getLogger("pycomposer")
        self.__logger = logger
    
    def compute(self, x):
        '''
        Compute KL divergence as a cost.
        
        Args:
            x:    `np.ndarray` of explanatory variables.
        
        Returns:
            cost
        '''
        self.__epoch += 1
        # Inference feature points.
        inferenced_arr = self.__rbm.inference(
            x,
            training_count=1, 
            r_batch_size=self.__batch_size
        )

        cost = np.sum(
            (inferenced_arr + 1e-08) * np.log((inferenced_arr + 1e-08) / (self.__inferenced_arr + 1e-08)),
            axis=(inferenced_arr.ndim - 1)
        )
        
        self.__logger.debug("Epoch: " + str(self.__epoch) + " Cost: " + str(cost.mean()))

        return cost.mean()
