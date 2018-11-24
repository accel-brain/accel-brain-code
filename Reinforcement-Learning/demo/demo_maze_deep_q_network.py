# -*- coding: utf-8 -*-
from pyqlearning.functionapproximator.cnn_fa import CNNFA
from pyqlearning.deepqlearning.deep_q_network import DeepQNetwork
import numpy as np


class MazeDeepQNetwork(DeepQNetwork):
    '''
    Deep Q-Network to solive Maze problem.
    '''

    def __init__(self, function_approximator, map_size=(10, 10)):
        '''
        Init.
        
        Args:
            function_approximator:  is-a `FunctionApproximator`.
        '''
        self.__map_arr = np.random.uniform(low=0, high=1, size=map_size)
        self.__map_arr[0, 0] = -1
        self.__map_arr[-1, -1] = 1
        self.__agent_pos = (0, 0)
        
        self.__reward_list = []
        
        super().__init__(function_approximator)
    
    def inference(self, state_arr, limit=1000):
        '''
        Infernce.
        
        Args:
            state_arr:    `np.ndarray` of state.
            limit:        The number of inferencing.
        
        Returns:
            `list of `np.ndarray` of an optimal route.
        '''
        result_list = [state_arr]
        self.t = 1
        while self.t <= limit:
            next_action_arr = self.extract_possible_actions(state_arr)
            next_q_arr = self.function_approximator.inference_q(next_action_arr)
            action_arr, q = self.select_action(next_action_arr, next_q_arr)

            result_list.append(action_arr)

            # Update State.
            state_arr = self.update_state(state_arr, action_arr)

            # Epsode.
            self.t += 1
            # Check.
            end_flag = self.check_the_end_flag(state_arr)
            if end_flag is True:
                break

        return result_list

    def extract_possible_actions(self, state_arr):
        '''
        Extract possible actions.

        Args:
            state_arr:  `np.ndarray` of state.
        
        Returns:
            `np.ndarray` of actions.
            The shape is:(
                `batch size corresponded to each action key`, 
                `channel that is 1`, 
                `feature points1`, 
                `feature points2`
            )
        '''
        agent_x, agent_y = np.where(state_arr[0] == 1)
        agent_x, agent_y = agent_x[0], agent_y[0]

        possible_action_arr = None
        for x in [-1, 0, 1]:
            next_x = agent_x + x
            if next_x < 0 or next_x >= state_arr[0].shape[1]:
                continue
            for y in [-1, 0, 1]:
                next_y = agent_y + y
                if next_y < 0 or next_y >= state_arr[0].shape[0]:
                    continue
                if x == 0 and y == 0:
                    continue

                next_action_arr = np.zeros((1, state_arr[0].shape[0], state_arr[0].shape[1]))
                next_action_arr[0][next_x, next_y] = 1
                next_action_arr = np.expand_dims(next_action_arr, axis=0)
                if possible_action_arr is None:
                    possible_action_arr = next_action_arr
                else:
                    possible_action_arr = np.r_[possible_action_arr, next_action_arr]

        while possible_action_arr.shape[0] < 8:
            key = np.random.randint(low=0, high=possible_action_arr.shape[0])
            possible_action_arr = np.r_[
                possible_action_arr,
                np.expand_dims(possible_action_arr[key], axis=0)
            ]
        return possible_action_arr
    
    def observe_reward_value(self, state_arr, action_arr):
        '''
        Compute the reward value.
        
        Args:
            state_arr:              `np.ndarray` of state.
            action_arr:             `np.ndarray` of action.
        
        Returns:
            Reward value.
        '''
        x, y = np.where(action_arr[0] == 1)
        self.__agent_pos = (x[0], y[0])
        
        self.__reward_list.append(self.__map_arr[x[0], y[0]])
        return self.__map_arr[x[0], y[0]]

    def extract_now_state(self):
        '''
        Extract now map state.
        
        Returns:
            `np.ndarray` of state.
        '''
        x, y = self.__agent_pos
        state_arr = np.zeros(self.__map_arr.shape)
        state_arr[x, y] = 1
        return np.expand_dims(state_arr, axis=0)

    def update_state(self, state_arr, action_arr):
        '''
        Update state.
        
        Override.

        Args:
            state_arr:    `np.ndarray` of state in `self.t`.
            action_arr:   `np.ndarray` of action in `self.t`.
        
        Returns:
            `np.ndarray` of state in `self.t+1`.
        '''
        x, y = np.where(action_arr[0] == 1)
        self.__agent_pos = (x[0], y[0])
        return self.extract_now_state()

    def check_the_end_flag(self, state_arr):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_arr:    `np.ndarray` of state in `self.t`.

        Returns:
            bool
        '''
        x, y = np.where(state_arr[0] == 1)
        if x[0] == self.__map_arr.shape[0] - 1 and y[0] == self.__map_arr.shape[1] - 1:
            print("Goal!")
            return True
        else:
            return False

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_map_arr(self):
        ''' getter '''
        return self.__map_arr
    
    map_arr = property(get_map_arr, set_readonly)
    
    def get_reward_list(self):
        ''' getter '''
        return self.__reward_list

    reward_list = property(get_reward_list, set_readonly)