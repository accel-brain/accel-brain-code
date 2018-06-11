# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import warnings


class AnnealingModel(metaclass=ABCMeta):
    '''
    Abstract class of Annealing.
    '''
    # The fitted data points.
    __var_arr = None

    # The np.ndarray of the log of predicted score.
    __predicted_log_arr = None

    # Optimized data points.
    __var_log_arr = None

    # The `np.ndarray` of computed cost.
    __computed_cost_arr = None

    def fit_dist_mat(self, dist_mat_arr):
        '''
        Fit ovserved data points.

        Args:
            dist_mat_arr:    fitted data points.
        '''
        warnings.warn("This property will be removed in future version. Use `var_arr`.", FutureWarning)
        # Set the data points.
        self.var_arr = dist_mat_arr
        
    @abstractmethod
    def annealing(self):
        '''
        Annealing.
        '''
        raise NotImplementedError()

    def get_var_arr(self):
        ''' getter '''
        if isinstance(self.__var_arr, np.ndarray):
            return self.__var_arr
        else:
            raise TypeError()
    
    def set_var_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__var_arr = value
        else:
            raise TypeError()

    var_arr = property(get_var_arr, set_var_arr)

    def get_predicted_log_arr(self):
        ''' getter '''
        if isinstance(self.__predicted_log_arr, np.ndarray):
            return self.__predicted_log_arr
        else:
            raise TypeError()
    
    def set_predicted_log_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__predicted_log_arr = value
        else:
            raise TypeError()
    
    predicted_log_arr = property(get_predicted_log_arr, set_predicted_log_arr)

    def get_var_log_arr(self):
        ''' getter '''
        if isinstance(self.__var_log_arr, np.ndarray):
            return self.__var_log_arr
        else:
            raise TypeError()
    
    def set_var_log_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__var_log_arr = value
        else:
            raise TypeError()

    var_log_arr = property(get_var_log_arr, set_var_log_arr)

    def get_computed_cost_arr(self):
        ''' getter '''
        if isinstance(self.__computed_cost_arr, np.ndarray):
            return self.__computed_cost_arr
        else:
            raise TypeError()
    
    def set_computed_cost_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__computed_cost_arr = value
        else:
            raise TypeError()
    
    computed_cost_arr = property(get_computed_cost_arr, set_computed_cost_arr)


#########################################################################################################    
# Removed in future version.
#########################################################################################################    

    def get_predicted_log_list(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `predicted_log_arr`.", FutureWarning)
        return self.predicted_log_arr

    def set_predicted_log_list(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `predicted_log_arr`.", FutureWarning)
        raise TypeError("This property must be read-only.")

    predicted_log_list = property(get_predicted_log_list, set_predicted_log_list)

    def get_x(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr`.", FutureWarning)
        return self.var_log_arr

    def set_x(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr`.", FutureWarning)
        self.var_log_arr = value

    x = property(get_x, set_x)

    def get_stocked_predicted_arr(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr`.", FutureWarning)
        return self.computed_cost_arr

    def set_stocked_predicted_arr(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr`.", FutureWarning)
        raise TypeError("This property must be read-only.")

    stocked_predicted_arr = property(get_stocked_predicted_arr, set_stocked_predicted_arr)

    def get_current_dist_arr(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr[-1]`.", FutureWarning)
        return self.var_log_arr[-1]

    def set_current_dist_arr(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr`.", FutureWarning)
        raise TypeError("This property must be read-only.")

    current_dist_arr = property(get_current_dist_arr, set_current_dist_arr)

    def get_current_cost_arr(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr[-1]`.", FutureWarning)
        return self.computed_cost_arr[-1]

    def set_current_cost_arr(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr[-1]`.", FutureWarning)
        raise TypeError("This property must be read-only.")
    
    current_cost_arr = property(get_current_cost_arr, set_current_cost_arr)
    
    def get_accepted_pos(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version.", FutureWarning)
        return None

    def set_accepted_pos(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version.", FutureWarning)
        raise TypeError("This property must be read-only.")

    accepted_pos = property(get_accepted_pos, set_accepted_pos)
