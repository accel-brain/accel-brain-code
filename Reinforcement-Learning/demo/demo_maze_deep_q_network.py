# -*- coding: utf-8 -*-
from pyqlearning.functionapproximator.cnn_fa import CNNFA
from pyqlearning.deepqlearning.deep_q_network import DeepQNetwork
import numpy as np


class MazeDeepQNetwork(DeepQNetwork):
    '''
    Deep Q-Network to solive Maze problem.
    '''
    
    SPACE = 1
    WALL = -1
    START = 0
    GOAL = 3
    
    START_POS = (1, 1)
    
    __route_memory_list = []
    __route_long_memory_list = []
    __memory_num = 4

    def __init__(self, function_approximator, map_size=(10, 10), memory_num=4, repeating_penalty=0.5):
        '''
        Init.
        
        Args:
            function_approximator:  is-a `FunctionApproximator`.
            map_size:               Size of map.
            memory_num:             The number of step of agent's memory.
            repeating_penalty:      The value of penalty in the case that agent revisit.

        '''
        self.__map_arr = self.__create_map(map_size)
        self.__agent_pos = self.START_POS
        
        self.__reward_list = []
        self.__route_memory_list = []
        self.__memory_num = memory_num
        self.__repeating_penalty = repeating_penalty

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
        agent_x, agent_y = np.where(state_arr[0] == 1)
        agent_x, agent_y = agent_x[0], agent_y[0]
        result_list = [(agent_x, agent_y, 0.0)]
        self.t = 1
        while self.t <= limit:
            next_action_arr = self.extract_possible_actions(state_arr)
            next_q_arr = self.function_approximator.inference_q(next_action_arr)
            action_arr, q = self.select_action(next_action_arr, next_q_arr)

            agent_x, agent_y = np.where(action_arr[0] == 1)
            agent_x, agent_y = agent_x[0], agent_y[0]
            result_list.append((agent_x, agent_y, q[0]))

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
        agent_x, agent_y = np.where(state_arr[-1] == 1)
        agent_x, agent_y = agent_x[0], agent_y[0]

        possible_action_arr = None
        for x, y in [
            (-1, 0), (1, 0), (0, -1), (0, 1)
        ]:
            next_x = agent_x + x
            if next_x < 0 or next_x >= state_arr[-1].shape[1]:
                continue
            next_y = agent_y + y
            if next_y < 0 or next_y >= state_arr[-1].shape[0]:
                continue

            wall_flag = False
            if x > 0:
                for add_x in range(1, x):
                    if self.__map_arr[agent_x + add_x, next_y] == self.WALL:
                        wall_flag = True
            elif x < 0:
                for add_x in range(x, 0):
                    if self.__map_arr[agent_x + add_x, next_y] == self.WALL:
                        wall_flag = True
                    
            if wall_flag is True:
                continue

            if y > 0:
                for add_y in range(1, y):
                    if self.__map_arr[next_x, agent_y + add_y] == self.WALL:
                        wall_flag = True
            elif y < 0:
                for add_y in range(y, 0):
                    if self.__map_arr[next_x, agent_y + add_y] == self.WALL:
                        wall_flag = True

            if wall_flag is True:
                continue

            if self.__map_arr[next_x, next_y] == self.WALL:
                continue

            if (next_x, next_y) in self.__route_memory_list:
                continue

            next_action_arr = np.zeros((3, state_arr[-1].shape[0], state_arr[-1].shape[1]))
            next_action_arr[0][agent_x, agent_y] = 1
            next_action_arr[1] = self.__map_arr
            next_action_arr[2][next_x, next_y] = 1
            next_action_arr = np.expand_dims(next_action_arr, axis=0)
            if possible_action_arr is None:
                possible_action_arr = next_action_arr
            else:
                possible_action_arr = np.r_[possible_action_arr, next_action_arr]

        if possible_action_arr is not None:
            while possible_action_arr.shape[0] < 4:
                key = np.random.randint(low=0, high=possible_action_arr.shape[0])
                possible_action_arr = np.r_[
                    possible_action_arr,
                    np.expand_dims(possible_action_arr[key], axis=0)
                ]
        else:
            # Forget oldest memory and do recuresive executing.
            self.__route_memory_list = self.__route_memory_list[1:]
            possible_action_arr = self.extract_possible_actions(state_arr)

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
        if self.__check_goal_flag(action_arr) is True:
            return 1.0
        else:
            x, y = np.where(action_arr[-1] == 1)
            x, y = x[0], y[0]
            goal_x, goal_y = self.__goal_pos
            
            if x == goal_x and y == goal_y:
                distance = 0.0
            else:
                distance = np.sqrt(((x - goal_x) ** 2) + (y - goal_y) ** 2)

            if (x, y) in self.__route_long_memory_list:
                repeating_penalty = self.__repeating_penalty
            else:
                repeating_penalty = 0.0
            return 1.0 - distance - repeating_penalty

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
        x, y = np.where(action_arr[-1] == 1)
        self.__agent_pos = (x[0], y[0])
        self.__route_memory_list.append((x[0], y[0]))
        self.__route_long_memory_list.append((x[0], y[0]))
        self.__route_long_memory_list = list(set(self.__route_long_memory_list))
        while len(self.__route_memory_list) > self.__memory_num:
            self.__route_memory_list = self.__route_memory_list[1:]

        return self.extract_now_state()

    def __check_goal_flag(self, state_arr):
        x, y = np.where(state_arr[0] == 1)
        goal_x, goal_y = self.__goal_pos
        if x[0] == goal_x and y[0] == goal_y:
            return True
        else:
            return False
    
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
        if self.__check_goal_flag(state_arr) is True:
            return True
        else:
            return False

    def __create_map(self, map_size):
        '''
        Create map.
        
        References:
            - https://qiita.com/kusano_t/items/487eec15d42aace7d685
        '''
        import random
        import numpy as np
        from itertools import product

        news = ['n', 'e', 'w', 's']

        m, n = map_size

        SPACE = self.SPACE
        WALL = self.WALL
        START = self.START
        GOAL = self.GOAL

        memo = np.array([i for i in range(n * m)])
        memo = memo.reshape(m, n)

        # 迷路を初期化
        maze = [[SPACE for _ in range(2 * n + 1)] for _ in range(2 * m + 1)]
        maze[self.START_POS[0]][self.START_POS[1]] = START
        
        self.__goal_pos = (2 * m - 1, 2 * n - 1)

        maze[2 * m - 1][2 * n - 1] = GOAL
        for i, j in product(range(2 * m + 1), range(2 * n + 1)):
            if i % 2 == 0 or j % 2 == 0:
                maze[i][j] = WALL

        while (memo != 0).any():
            x1 = random.choice(range(m))
            y1 = random.choice(range(n))
            direction = random.choice(news)

            if direction == 'e':
                x2, y2 = x1, y1 + 1
            elif direction == 'w':
                x2, y2 = x1, y1 - 1
            elif direction == 'n':
                x2, y2 = x1 - 1, y1
            elif direction == 's':
                x2, y2 = x1 + 1, y1

            # 範囲外の場合はcontinue
            if (x2 < 0) or (x2 >= m) or (y2 < 0) or (y2 >= n):
                continue

            if memo[x1, y1] != memo[x2, y2]:
                tmp_min = min(memo[x1, y1], memo[x2, y2])
                tmp_max = max(memo[x1, y1], memo[x2, y2])

                # メモの更新
                memo[memo == tmp_max] = tmp_min

                # 壁を壊す
                maze[x1 + x2 + 1][y1 + y2 + 1] = SPACE

        maze_arr = np.array(maze)
        return maze_arr

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