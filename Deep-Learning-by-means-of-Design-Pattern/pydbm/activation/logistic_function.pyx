# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class LogisticFunction(ActivatingFunctionInterface):
    '''
    Logistic Function.
    '''

    def activate(self, double x):
        '''
        Return of result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, double y):
        '''
        Derivative.

        Args:
            y:  Parameter.
        Returns:
            The result.
        '''
        return self.activate(y) * (1 - self.activate(y))
