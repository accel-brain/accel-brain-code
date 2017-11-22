#!/user/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
from pyqlearning.qlearning.greedy_q_learning import GreedyQLearning


class MazeGreedyQLearning(GreedyQLearning):
    '''
    ε-greedy Q-Learning

    Refererence:
        http://d.hatena.ne.jp/Kshi_Kshi/20111227/1324993576

    '''
    
    # Map of maze.
    __map_arr = None
    # Start point.
    __start_point_tuple = (0, 0)
    # End point.
    __end_point_tuple = (20, 20)
    # Label of start point.
    __start_point_label = "S"
    # Label of end point.
    __end_point_label = "G"
    # Label of wall.
    __wall_label = "#"
    # Label of agent.
    __agent_label = "@"

    # Map logs.
    __map_arr_list = []

    def initialize(self, map_arr, start_point_label="S", end_point_label="G", wall_label="#", agent_label="@"):
        '''
        Initialize map of maze and setup reward value.

        Args:
            map_arr:              Map. the 2d- `np.ndarray`.
            start_point_label:    Label of start point.
            end_point_label:      Label of end point.
            wall_label:           Label of wall.
            agent_label:          Label of agent.

        '''
        np.set_printoptions(threshold=np.inf)

        self.__agent_label = agent_label
        self.__map_arr = map_arr
        self.__start_point_label = start_point_label
        start_arr_tuple = np.where(self.__map_arr == self.__start_point_label)
        x_arr, y_arr = start_arr_tuple
        self.__start_point_tuple = (x_arr[0], y_arr[0])
        end_arr_tuple = np.where(self.__map_arr == self.__end_point_label)
        x_arr, y_arr = end_arr_tuple
        self.__end_point_tuple = (x_arr[0], y_arr[0])
        self.__wall_label = wall_label

        for x in range(self.__map_arr.shape[1]):
            for y in range(self.__map_arr.shape[0]):
                if (x, y) == self.__start_point_tuple or (x, y) == self.__end_point_tuple:
                    continue
                arr_value = self.__map_arr[y][x]
                if arr_value == self.__wall_label:
                    continue
                    
                self.save_r_dict((x, y), float(arr_value))

    def extract_possible_actions(self, state_key):
        '''
        Concreat method.

        Args:
            state_key       The key of state. this value is point in map.

        Returns:
            [(x, y)]

        '''
        x, y = state_key
        if self.__map_arr[y][x] == self.__wall_label:
            raise ValueError("It is the wall. (x, y)=(%d, %d)" % (x, y))

        around_map = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        possible_actoins_list = [(_x, _y) for _x, _y in around_map if self.__map_arr[_y][_x] != self.__wall_label and self.__map_arr[_y][_x] != self.__start_point_label]
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
        x, y = state_key

        if self.__map_arr[y][x] == self.__end_point_label:
            return 100.0
        elif self.__map_arr[y][x] == self.__start_point_label:
            return 0.0
        elif self.__map_arr[y][x] == self.__wall_label:
            raise ValueError("It is the wall. (x, y)=(%d, %d)" % (x, y))
        else:
            reward_value = float(self.__map_arr[y][x])
            self.save_r_dict(state_key, reward_value)
            return reward_value

    def visualize_learning_result(self, state_key):
        '''
        Visualize learning result.
        '''
        x, y = state_key
        map_arr = copy.deepcopy(self.__map_arr)
        goal_point_tuple = np.where(map_arr == self.__end_point_label)
        goal_x, goal_y = goal_point_tuple
        map_arr[y][x] = "@"
        self.__map_arr_list.append(map_arr)
        if goal_x == x and goal_y == y:
            for i in range(10):
                key = len(self.__map_arr_list) - (10 - i)
                print("Number of searches: " + str(key))
                print(self.__map_arr_list[key])
            print("Total number of searches: " + str(self.t))
            print(self.__map_arr_list[-1])
            print("Goal !!")

    def check_the_end_flag(self, state_key):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        Args:
            state_key:    The key of state in `self.t`.

        Returns:
            bool
        '''
        # As a rule, the learning can not be stopped.
        x, y = state_key
        end_point_tuple = np.where(self.__map_arr == self.__end_point_label)
        end_point_x_arr, end_point_y_arr = end_point_tuple
        if x == end_point_x_arr[0] and y == end_point_y_arr[0]:
            return True
        else:
            return False
