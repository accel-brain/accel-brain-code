# -*- coding: utf-8 -*-
import random
import numpy as np
import copy
from pyqlearning.qlearning.greedy_q_learning import GreedyQLearning


class DungeonGreedyQLearning(GreedyQLearning):
    
    __map_arr = None
    __move_range = 1
    
    def __init__(self, map_arr, now_pos_tuple, move_range):
        self.__map_arr = map_arr
        self.__now_pos_tuple = now_pos_tuple
        self.__move_range = move_range
    
    def extract_possible_actions(self, state_key):
        '''
        Concreat method.

        Args:
            state_key       The key of state. this value is point in map.

        Returns:
            [(x, y)]

        '''
        my_key, state_key_list = state_key
        my_state_tuple = state_key_list[my_key]
        
        possible_actoins_list = []
        for move_y in range(-self.__mvoe_range, self.__move_range + 1):
            for move_x in range(-self.__mvoe_range, self.__move_range + 1):
                try:
                    next_pos = self.__map_arr[my_state_tuple[0] + move_y, my_state_tuple[1] + move_x]
                    if next_pos == -1:
                        continue
                    possible_actoins_list.append((my_key, (my_state_tuple[0] + move_y, my_state_tuple[1] + move_x)))
                except IndexError:
                    continue

        return possible_actoins_list

    def observe_reward_value(self, state_key, action_key):
        '''
        Compute the reward value.
        
        Args:
            state_key:              The key of state.
            action_key:             The key of action.
        
        Returns:
            Reward value.

        '''
        my_key, state_key_list = state_key
        my_state_tuple = state_key_list[my_key]

        my_key_, action_key_list = action_key
        my_action_tuple = action_key_list[my_key_]
        
        pos = self.__map_arr[my_action_tuple[0], my_action_tuple[1]]
        # Goal.
        if pos == 1:
            return 100.0
        elif pos == -1:
            return 0.0
        else:
            reward_value = 0.0
            for i in range(len(state_key_list)):
                if i == my_key:
                    continue
                # Hit.
                if pos == state_key_list[i]:
                    reward_value = reward_value - 10.0

            self.save_r_df(state_key, reward_value, my_action_tuple)
            return reward_value

    '''
    def update_q(self, state_key, action_key, reward_value, next_max_q):
        my_key, state_key_list = state_key
        my_state_tuple = state_key_list[my_key]

        my_key_, action_key_list = action_key
        my_action_tuple = action_key_list[my_key_]
        
        super().update_q(my_state_tuple, my_action_tuple, reward_value, next_max_q)
    '''

    def check_the_end_flag(self, state_key):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_key:    The key of state in `self.t`.

        Returns:
            bool
        '''
        # As a rule, the learning can not be stopped.
        my_key, state_key_list = state_key
        my_state_tuple = state_key_list[my_key]
        pos = self.__map_arr[my_state_tuple[0], my_state_tuple[1]]
        
        if pos == 1:
            return True
        else:
            return False
