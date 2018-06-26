# -*- coding: utf-8 -*-
from pyqlearning.q_learning import QLearning
import pandas as pd
import numpy as np
import random


class MultiAgentQLearning(object):
    '''
    Controler for Multi Agent Q-Learning.

    Attributes:
        alpha_value:        Learning rate.
        gamma_value:        Gammma value.
        q_dict:             Q(state, action) 
        r_dict:             R(state)
        t:                  time.

    '''
    
    __q_learning_list = []
    
    def get_q_learning_list(self):
        return self.__q_learning_list
    
    def set_readonly(self, value):
        raise TypeError()
    
    q_learning_list = property(get_q_learning_list, set_readonly)
    
    __state_key_list = []
    
    def get_state_key_list(self):
        return self.__state_key_list

    def set_state_key_list(self, value):
        self.__state_key_list = value

    state_key_list = property(get_state_key_list, set_state_key_list)

    def __init__(self, q_learning_list):
        '''
        '''
        for q_learinng in q_learning_list:
            if isinstance(q_learinng, QLearning) is False:
                raise TypeError()

        self.__q_learning_list = q_learning_list

    # Time.
    __t = 0

    def get_t(self):
        '''
        getter
        Time.
        '''
        if isinstance(self.__t, int) is False:
            raise TypeError("The type of __t must be int.")
        return self.__t

    def set_t(self, value):
        '''
        setter
        Time.
        '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __t must be int.")
        self.__t = value

    t = property(get_t, set_t)

    def learning_interaction(self, first_state_key, limit=1000, game_n=1):
        end_flag = False
        for game in range(game_n):
            state_key = first_state_key
            self.t = 1
            while self.t <= limit:
                for i in range(len(self.__q_learning_list)):
                    if game + 1 == game_n:
                        self.state_key_list.append(state_key)
                    self.__q_learning_list[i].t = self.t
                    next_action_list = self.__q_learning_list[i].extract_possible_actions(state_key)
                    if len(next_action_list):
                        action_key = self.__q_learning_list[i].select_action(
                            state_key=state_key,
                            next_action_list=next_action_list
                        )
                        reward_value = self.__q_learning_list[i].observe_reward_value(state_key, action_key)

                        # Check.
                        if self.__q_learning_list[i].check_the_end_flag(state_key) is True:
                            end_flag = True

                        # Max-Q-Value in next action time.
                        next_next_action_list = self.__q_learning_list[i].extract_possible_actions(action_key)
                        if len(next_next_action_list):
                            next_action_key = self.__q_learning_list[i].predict_next_action(action_key, next_next_action_list)
                            next_max_q = self.__q_learning_list[i].extract_q_df(action_key, next_action_key)

                            # Update Q-Value.
                            self.__q_learning_list[i].update_q(
                                state_key=state_key,
                                action_key=action_key,
                                reward_value=reward_value,
                                next_max_q=next_max_q
                            )

                            # Update State.
                            state_key = self.__q_learning_list[i].update_state(
                                state_key=state_key,
                                action_key=action_key
                            )

                    # Epsode.
                    self.t += 1
                    self.__q_learning_list[i].t = self.t
                    if end_flag is True:
                        break
