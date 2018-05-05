# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from pyqlearning.qlearning.greedy_q_learning import GreedyQLearning
from pycomposer.inferable_consonance import InferableConsonance


class QConsonance(GreedyQLearning, InferableConsonance):
    
    # Default `r_dict`.
    __default_r_dict = {
        (1, 1): 1.0,
        (15, 16): 0.0,
        (8, 9): 0.0,
        (5, 6): 0.5,
        (4, 5): 0.5,
        (2, 1): 1.0,
        (32, 45): 0.0,
        (2, 3): 1.0,
        (5, 8): 0.5,
        (3, 5): 0.5,
        (9, 16): 0.0,
        (8, 15): 0.0,
        (1, 2): 1.0
    }
    
    __r_dict = {}
    
    __tone_df = None
    
    def initialize(self, r_dict=None, tone_df=None):
        '''
        Initialize the definition of Consonances.

        Args:
        '''
        if r_dict is None:
            r_dict = self.__default_r_dict

        for key, val in r_dict.items():
            pre, post = key
            self.__r_dict.setdefault((float(pre), float(post)), val)
        
        self.__tone_df = tone_df

    def extract_possible_actions(self, state_key):
        '''
        Concreat method.

        Args:
            state_key       The key of state.
        Returns:
            [pitch]
        '''
        def extract(state_key):
            next_arr = np.arange(state_key-12, state_key+12)
            next_arr = next_arr[next_arr >= 0]
            next_arr = next_arr[next_arr <= 127]
            return next_arr

        if self.__tone_df is None:
            next_arr = extract(state_key)
        else:
            next_arr = self.__tone_df.pitch.drop_duplicates().values
            next_arr = next_arr[next_arr >= state_key - 12]
            next_arr = next_arr[next_arr <= state_key + 12]
            if next_arr.shape[0] == 0:
                next_arr = extract(state_key)

        return next_arr.tolist()

    def __gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def observe_reward_value(self, state_key, action_key):
        '''
        Compute the reward value.
        
        Args:
            state_key:              The key of state.
            action_key:             The key of action.
        
        Returns:
            Reward value.
        '''
        pre_f = self.__frequency_from_pitch(state_key)
        post_f = self.__frequency_from_pitch(action_key)
        
        g = self.__gcd(pre_f, post_f)
        
        pre_r = pre_f/g
        post_r = post_f/g
        
        if (pre_r, post_r) in self.__r_dict:
            reward_value = self.__r_dict[(pre_r, post_r)]
        else:
            reward_value = 0.0

        self.save_r_df(state_key, reward_value, action_key)
        return reward_value

    def __frequency_from_pitch(self, pitch):
        return 440 * (2 ** ((pitch - 69) / 12))

    def visualize_learning_result(self, state_key):
        '''
        Visualize learning result.
        '''
        pass

    def check_the_end_flag(self, state_key):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.
        Args:
            state_key:    The key of state in `self.t`.
        Returns:
            bool
        '''
        return False

    def normalize_q_value(self):
        '''
        Normalize q-value.
        
        Override.
        
        This method is called in each learning steps.
        
        For example:
            self.q_df.q_value = self.q_df.q_value / self.q_df.q_value.sum()
        '''
        if self.q_df is not None and self.q_df.shape[0]:
            # min-max normalization
            self.q_df.q_value = (self.q_df.q_value - self.q_df.q_value.min()) / (self.q_df.q_value.max() - self.q_df.q_value.min())

    def normalize_r_value(self):
        '''
        Normalize r-value.
        Override.
        This method is called in each learning steps.
        For example:
            self.r_df = self.r_df.r_value / self.r_df.r_value.sum()
        '''
        if self.r_df is not None and self.r_df.shape[0]:
            # z-score normalization.
            self.r_df.r_value = (self.r_df.r_value - self.r_df.r_value.mean()) / self.r_df.r_value.std()

    def inference(self, pre_pitch, limit=5):
        '''
        Inference the degree of Consonance.
        
        Args:
            pre_pitch:    The pitch in `t-1`.
            limit:        The number of return list.
        
        Returns:
            The list of pitchs in `t`.
        '''
        q_df = self.q_df[self.q_df.state_key == pre_pitch]
        q_df = q_df.sort_values(by=["q_value"], ascending=False)
        return q_df.action_key.values[:limit]
