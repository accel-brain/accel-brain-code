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
    
    def __init__(self, q_learning_list):
        '''
        '''
        for q_learinng in q_learning_list:
            if isinstance(q_learning, QLearning) is False:
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

    def learn(self, state_key_list=None, limit=1000, end_condition="AllClear", agents_info="share"):
        '''
        Learning.
        '''
        self.t = 1
        if state_key_list is None:
            state_key_list = [None] * len(self.__q_learning_list)
        else:
            if isinstance(state_key_list, list) is False:
                raise TypeError()
            elif len(state_key_list) != len(self.__q_learning_list):
                raise ValueError()
        action_key_list = [None] * len(self.__q_learning_list)
        next_action_key_list = [None] * len(self.__q_learning_list)
        end_flag_list = [False] * len(self.__q_learning_list)
        while self.t <= limit:
            for i in range(len(self.__q_learning_list)):
                if agent_info == "unknown":
                    state_key = state_key_list[i]
                    next_action_list = self.__q_learning_list[i].extract_possible_actions(state_key)
                    action_key = self.__q_learning_list[i].select_action(
                        state_key=state_key,
                        next_action_list=next_action_list
                    )
                    reward_value = self.__q_learning_list[i].observe_reward_value(state_key, action_key)

                    # Check.
                    if self.__q_learning_list[i].check_the_end_flag(state_key) is True:
                        end_flag_list[i] = True

                    # Max-Q-Value in next action time.
                    next_next_action_list = self.__q_learning_list[i].extract_possible_actions(action_key)
                    next_action_key = self.__q_learning_list[i].predict_next_action(action_key, next_next_action_list)
                    next_max_q = self.__q_learning_list[i].extract_q_df(action_key, next_action_key)

                    # Update Q-Value.
                    self.__q_learning_list[i].update_q(
                        state_key=state_key,
                        action_key=action_key,
                        reward_value=reward_value,
                        next_max_q=next_max_q
                    )

                    # Epsode.
                    self.t += 1
                    self.__q_learning_list[i].t = self.t

                    # Update State.
                    state_key = self.__q_learning_list[i].update_state(
                        state_key=state_key,
                        action_key=action_key
                    )

                    state_key_list[i] = state_key
                else:
                    state_key = state_key_list[i]
                    next_action_list = self.__q_learning_list[i].extract_possible_actions((i, tuple(state_key_list)))
                    action_key = self.__q_learning_list[i].select_action(
                        state_key=(i, tuple(state_key_list)),
                        next_action_list=next_action_list
                    )
                    action_key_list[i] = action_key
                    reward_value = self.__q_learning_list[i].observe_reward_value(
                        (i, tuple(state_key_list)),
                        (i, tuple(action_key_list))
                    )

                    # Check.
                    if self.__q_learning_list[i].check_the_end_flag((i, tuple(state_key_list))) is True:
                        end_flag_list[i] = True

                    # Max-Q-Value in next action time.
                    next_next_action_list = self.__q_learning_list[i].extract_possible_actions((i, tuple(action_key_list)))
                    next_action_key = self.__q_learning_list[i].predict_next_action(
                        (i, tuple(action_key_list)),
                        next_next_action_list
                    )
                    next_action_key_list[i] = next_action_key
                    next_max_q = self.__q_learning_list[i].extract_q_df(
                        (i, tuple(action_key_list)), 
                        (i, tuple(next_action_key_list))
                    )

                # Normalize.
                self.__q_learning_list[i].normalize_q_value()
                self.__q_learning_list[i].normalize_r_value()

            if end_condition == "AllClear":
                if False not in end_flag_list:
                    break
            elif end_condition == "FirstClear":
                if True in end_flag_list:
                    break
