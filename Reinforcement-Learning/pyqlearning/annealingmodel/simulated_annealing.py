# -*- coding: utf-8 -*-
import numpy as np
from pyqlearning.annealing_model import AnnealingModel
from pyqlearning.annealingmodel.cost_functionable import CostFunctionable


class SimulatedAnnealing(AnnealingModel):
    '''
    Simulated Annealing.
    '''
    # Number of cycles.
    __cycles_num = 200
    # Number of trials per cycle.
    __trials_per_cycle = 50
    # Number of accepted solutions.
    __accepted_sol_num = 0.0
    # Probability of accepting worse solution at the start.
    __init_prob = 0.7
    # Probability of accepting worse solution at the end.
    __final_prob = 0.001

    # Start location
    __start_pos = 0

    # Initial temperature
    __init_temp = None
    # Final temperature
    __final_temp = None

    # User function for optimization
    __cost_functionable = None

    # The scalar of fractional reduction in every cycle
    # for lower the temperature for next cycle
    __fractional_reduction = None
    
    # Range of moving in the feature map.
    __move_range = 3

    def __init__(
        self,
        cost_functionable,
        cycles_num=200,
        trials_per_cycle=50,
        accepted_sol_num=0.0,
        init_prob=0.7,
        final_prob=0.001,
        start_pos=0,
        move_range=3,
    ):
        '''
        Initialize.

        Args:
            cost_functionalbe:    The object of `CostFunctionable`.
            cycles_num:           The number of annealing cycles.
            trials_per_cycles:    The number of traials per the cycles.
            accepted_sol_num:     The number of acceptance solution.
            init_prob:            Probability of accepting worse solution at the start.
            final_prob:           Probability of accepting worse solution at the end.
            start_pos:            The first searched position.
            move_range:           The range of moving in the feature map.

        '''
        if isinstance(cost_functionable, CostFunctionable):
            self.__cost_functionable = cost_functionable
        else:
            raise TypeError('The type of `cost_functionable must be CostFunctionable.')

        self.__cycles_num = cycles_num
        self.__trials_per_cycle = trials_per_cycle
        self.__accepted_sol_num = accepted_sol_num + 1.0
        self.__init_prob = init_prob
        self.__final_prob = final_prob
        self.__start_pos = start_pos
        self.__move_range = move_range

        # Initial temperature
        self.__init_temp = -1.0/np.log(self.__init_prob)
        # Final temperature
        self.__final_temp = -1.0/np.log(self.__final_prob)
        # Fractional reduction every cycle.
        self.__fractional_reduction = (self.__final_temp / self.__init_temp) ** (1.0 / (self.__cycles_num-1.0))

    def __move(self, current_pos):
        '''
        Move in the feature map.

        Args:
            current_pos:    The now position.

        Returns:
            The next position.
        '''
        if self.__move_range is not None:
            next_pos = np.random.randint(current_pos - self.__move_range, current_pos + self.__move_range)
            if next_pos < 0:
                next_pos = 0
            elif next_pos >= self.dist_mat_arr.shape[0] - 1:
                next_pos = self.dist_mat_arr.shape[0] - 1
            return next_pos
        else:
            next_pos = np.random.randint(self.dist_mat_arr.shape[0] - 1)
            return next_pos

    def annealing(self):
        '''
        Annealing.
        '''
        self.x = np.zeros((self.__cycles_num + 1, self.dist_mat_arr.shape[1]))

        current_pos = self.__start_pos

        self.current_dist_arr = self.dist_mat_arr[current_pos, :]
        self.current_cost_arr = self.__cost_functionable.compute(self.dist_mat_arr[current_pos, :])
        self.stocked_predicted_arr = np.zeros(self.__cycles_num + 1)
        self.stocked_predicted_arr[0] = self.current_cost_arr

        t = self.__init_temp
        delta_e_avg = 0.0
        pos_log_list = [current_pos]
        predicted_log_list = []
        for i in range(self.__cycles_num):
            for j in range(self.__trials_per_cycle):
                current_pos = self.__move(current_pos)
                pos_log_list.append(current_pos)
                self.__now_dist_mat_arr = self.dist_mat_arr[current_pos, :]
                cost_arr = self.__cost_functionable.compute(self.__now_dist_mat_arr)
                delta_e = np.abs(cost_arr - self.current_cost_arr)
                
                if (cost_arr > self.current_cost_arr):
                    if (i == 0 and j == 0):
                        delta_e_avg = delta_e
                    p = np.exp(-delta_e/(delta_e_avg * t))
                    if (np.random.random() < p):
                        accept = True
                    else:
                        accept = False
                else:
                    accept = True
                    p = 0.0

                if accept is True:
                    self.current_dist_arr = self.__now_dist_mat_arr
                    self.current_cost_arr = cost_arr
                    self.__accepted_sol_num = self.__accepted_sol_num + 1.0
                    delta_e_avg = (delta_e_avg * (self.__accepted_sol_num - 1.0) +  delta_e) / self.__accepted_sol_num
                    self.accepted_pos = current_pos
                self.predicted_log_list.append((cost_arr , delta_e, delta_e_avg, p, int(accept)))
            self.x[i + 1] = self.current_dist_arr
            self.stocked_predicted_arr[i + 1] = self.current_cost_arr
            t = t * self.__fractional_reduction
