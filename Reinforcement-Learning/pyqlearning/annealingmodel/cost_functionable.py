# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class CostFunctionable(metaclass=ABCMeta):
    '''
    The interface of cost function in annealing.
    
    The definition of cost function is possible option: 
    not necessity but contingent from the point of view of modal logic.
    You should questions the necessity of definition and re-define, 
    for designing the implementation of this interface, 
    in relation to your problem settings.
    
    References:
        - Bertsimas, D., & Tsitsiklis, J. (1993). Simulated annealing. Statistical science, 8(1), 10-15.
        - Das, A., & Chakrabarti, B. K. (Eds.). (2005). Quantum annealing and related optimization methods (Vol. 679). Springer Science & Business Media.
        - Du, K. L., & Swamy, M. N. S. (2016). Search and optimization by metaheuristics. New York City: Springer.
        - Edwards, S. F., & Anderson, P. W. (1975). Theory of spin glasses. Journal of Physics F: Metal Physics, 5(5), 965.
    '''

    @abstractmethod
    def compute(self, x):
        '''
        Compute.

        Args:
            x:    var.
        
        Returns:
            Cost.
        '''
        raise NotImplementedError()
