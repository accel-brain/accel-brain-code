# -*- coding: utf-8 -*-
import random
import numpy as np
import copy
from pyqlearning.qlearning.greedy_q_learning import GreedyQLearning


class ReversiGreedyQLearning(GreedyQLearning):
    
    __color = 1

    def __init__(self, color):
        self.__color = color

    def extract_possible_actions(self, state_key):
        '''
        Concreat method.

        Args:
            state_key       The key of state. this value is point in map.

        Returns:
            [(x, y)]

        '''
        map_arr = state_key
        searched_arr_tuple = np.where(map_arr == 0)
        possible_y_x_list = np.c_[searched_arr_tuple[0], searched_arr_tuple[1]].tolist()
        possible_actoins_list = []
        for y_x_list in possible_y_x_list:
            pos_tuple = tuple(y_x_list)
            after_map_arr = self.__simulate(map_arr.copy(), pos_tuple, self.__color)
            diff_arr = (after_map_arr == map_arr).astype(int)
            if diff_arr[diff_arr == 0].shape[0] > 1:
                possible_actoins_list.append(after_map_arr)
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
        before_map_arr = state_key
        after_map_arr = action_key
        
        before_gain = before_map_arr[before_map_arr == self.__color].ravel().shape[0]
        after_gain = after_map_arr[after_map_arr == self.__color].ravel().shape[0]

        reward = (after_gain / before_gain) * self.t
        
        self.save_r_df(
            state_key=state_key,
            r_value=reward,
            action_key=action_key
        )

        return reward

    def extract_r_df(self, state_key, r_value, action_key=None):
        return super().extract_r_df(self.__tokenize(state_key), r_value, self.__tokenize(action_key))

    def save_r_df(self, state_key, r_value, action_key=None):
        super().save_r_df(self.__tokenize(state_key), r_value, self.__tokenize(action_key))

    def extract_q_df(self, state_key, action_key):
        '''
        Extract Q-Value from `self.q_dict`.
        Args:
            state_key:      The key of state.
            action_key:     The key of action.
        Returns:
            Q-Value.
        '''
        return super().extract_q_df(self.__tokenize(state_key), self.__tokenize(action_key))

    def save_q_df(self, state_key, action_key, q_value):
        '''
        Insert or update Q-Value in `self.q_dict`.
        Args:
            state_key:      State.
            action_key:     Action.
            q_value:        Q-Value.
        Exceptions:
            TypeError:      If the type of `q_value` is not float.
        '''
        super().save_q_df(self.__tokenize(state_key), self.__tokenize(action_key), q_value)

    def predict_next_action(self, state_key, next_action_list):
        '''
        Predict next action by Q-Learning.
        Args:
            state_key:          The key of state in `self.t+1`.
            next_action_list:   The possible action in `self.t+1`.
        Returns:
            The key of action.
        '''
        try:
            return super().predict_next_action(self.__tokenize(state_key), next_action_list)
        except TypeError:
            return next_action_list[0]
        except IndexError:
            return None

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
        if len(self.extract_possible_actions(state_key)) == 0:
            return True
        else:
            return False

    def __tokenize(self, arr):
        if isinstance(arr, np.ndarray):
            return "_".join(arr.ravel().astype(str).tolist())
        else:
            return arr

    def __simulate(self, map_arr, pos_tuple, color_key):
        map_arr[pos_tuple[0], pos_tuple[1]] = color_key

        possible_tuple_list = []
        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key = pos_tuple[1] - i
            if target_key < 0:
                break
            if map_arr[pos_tuple[0], target_key] == -1 * color_key:
                targeted_tuple_list.append((pos_tuple[0], target_key))
            else:
                if len(targeted_tuple_list):
                    if map_arr[pos_tuple[0], target_key] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key = pos_tuple[1] + i
            if target_key >= 7:
                break
            if map_arr[pos_tuple[0], target_key] == -1 * color_key:
                targeted_tuple_list.append((pos_tuple[0], target_key))
            else:
                if len(targeted_tuple_list):
                    if map_arr[pos_tuple[0], target_key] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key = pos_tuple[0] - i
            if target_key < 0:
                break
            if map_arr[target_key, pos_tuple[1]] == -1 * color_key:
                targeted_tuple_list.append((target_key, pos_tuple[1]))
            else:
                if len(targeted_tuple_list):
                    if map_arr[target_key, pos_tuple[1]] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key = pos_tuple[0] + i
            if target_key >= 7:
                break
            if map_arr[target_key, pos_tuple[1]] == -1 * color_key:
                targeted_tuple_list.append((target_key, pos_tuple[1]))
            else:
                if len(targeted_tuple_list):
                    if map_arr[target_key, pos_tuple[1]] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key_y = pos_tuple[0] - i
            target_key_x = pos_tuple[1] - i
            if target_key_x < 0:
                break
            if target_key_y < 0:
                break

            if map_arr[target_key_y, target_key_x] == -1 * color_key:
                targeted_tuple_list.append((target_key_y, target_key_x))
            else:
                if len(targeted_tuple_list):
                    if map_arr[target_key_y, target_key_x] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key_y = pos_tuple[0] + i
            target_key_x = pos_tuple[1] + i
            if target_key_y >= 7:
                break
            if target_key_x >= 7:
                break

            if map_arr[target_key_y, target_key_x] == -1 * color_key:
                targeted_tuple_list.append((target_key_y, target_key_x))
            else:
                if len(targeted_tuple_list):
                    if map_arr[target_key_y, target_key_x] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key_y = pos_tuple[0] - i
            target_key_x = pos_tuple[1] + i
            if target_key_y < 0:
                break
            if target_key_x >= 7:
                break

            if map_arr[target_key_y, target_key_x] == -1 * color_key:
                targeted_tuple_list.append((target_key_y, target_key_x))
            else:
                if len(targeted_tuple_list):
                    if map_arr[target_key_y, target_key_x] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        targeted_tuple_list = []
        reversed_flag = False
        for i in range(1, 7):
            target_key_y = pos_tuple[0] + i
            target_key_x = pos_tuple[1] - i
            if target_key_y >= 7:
                break
            if target_key_x < 0:
                break

            if map_arr[target_key_y, target_key_x] == -1 * color_key:
                targeted_tuple_list.append((target_key_y, target_key_x))
            else:
                if len(targeted_tuple_list):
                    if map_arr[target_key_y, target_key_x] == color_key:
                        for y, x in targeted_tuple_list:
                            possible_tuple_list.append((y, x))
                            map_arr[y, x] = color_key
                else:
                    break

        return map_arr
