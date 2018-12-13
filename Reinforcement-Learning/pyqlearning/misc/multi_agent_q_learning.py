# -*- coding: utf-8 -*-
from pyqlearning.q_learning import QLearning
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import random


class MultiAgentQLearning(metaclass=ABCMeta):
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
    
    def set_q_learning_list(self, value):
        self.__q_learning_list = value
    
    q_learning_list = property(get_q_learning_list, set_q_learning_list)
    
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
        self.state_key_list = []

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
    
    @abstractmethod
    def learn(self, initial_state_key, limit=1000, game_n=1):
        '''
        Multi-Agent Learning.

        Args:
            initial_state_key:  first state.
            limit:              Limit of the number of learning.
            game_n:             The number of games.
        '''
        raise NotImplementedError("This method must be implemented.")
