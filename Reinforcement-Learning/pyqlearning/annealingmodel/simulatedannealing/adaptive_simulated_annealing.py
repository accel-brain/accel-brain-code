# -*- coding: utf-8 -*-
import numpy as np
from pyqlearning.annealingmodel.simulated_annealing import SimulatedAnnealing
from pyqlearning.annealingmodel.cost_functionable import CostFunctionable


class AdaptiveSimulatedAnnealing(SimulatedAnnealing):
    '''
    Adaptive Simulated Annealing.
    
    Adaptive Simulated Annealing, also known as the 
    very fast simulated reannealing, is a very efficient 
    version of simulated annealing.
    
    References:
        - Bertsimas, D., & Tsitsiklis, J. (1993). Simulated annealing. Statistical science, 8(1), 10-15.
        - Du, K. L., & Swamy, M. N. S. (2016). Search and optimization by metaheuristics. New York City: Springer.
        - Mezard, M., & Montanari, A. (2009). Information, physics, and computation. Oxford University Press.
        - Nallusamy, R., Duraiswamy, K., Dhanalaksmi, R., & Parthiban, P. (2009). Optimization of non-linear multiple traveling salesman problem using k-means clustering, shrink wrap algorithm and meta-heuristics. International Journal of Nonlinear Science, 8(4), 480-487.

    '''

    # Now cycles.
    __now_cycles = 0
    # How often will this model reanneals there per cycles.
    __reannealing_per = 50
    # Thermostat.
    __thermostat = 0.9
    # The minimum temperature.
    __t_min = 0.001
    # The default temperature.
    __t_default = 1.0
    
    def adaptive_set(
        self,
        reannealing_per=50,
        thermostat=0.9,
        t_min=0.001,
        t_default=1.0
    ):
        '''
        Init for Adaptive Simulated Annealing.
        
        Args:
            reannealing_per:    How often will this model reanneals there per cycles.
            thermostat:         Thermostat.
            t_min:              The minimum temperature.
            t_default:          The default temperature.
        '''
        self.__reannealing_per = reannealing_per
        self.__thermostat = thermostat
        self.__t_min = t_min
        self.__t_default = t_default
    
    def change_t(self, t):
        '''
        Change temperature.
        
        Override.
        
        Args:
            t:    Now temperature.
        
        Returns:
            Next temperature.
        '''
        t = super().change_t(t)
        self.__now_cycles += 1
        if self.__now_cycles % self.__reannealing_per == 0:
            t = t * self.__thermostat
            
            if t < self.__t_min:
                t = self.__t_default
        return t
