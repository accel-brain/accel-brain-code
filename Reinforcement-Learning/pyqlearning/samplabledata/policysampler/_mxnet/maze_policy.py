# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.policy_sampler import PolicySampler

import mxnet.ndarray as nd
import mxnet as mx
import numpy as np


class MazePolicy(PolicySampler):
    '''
    Policy sampler for the multi-agent Deep Q-learning to evaluate the value of the "action".
    '''

    SPACE = 1
    WALL = -1
    START = 0
    GOAL = 3
    
    START_POS = (1, 1)
    
    __memory_num = 4

    END_STATE = "running"

    __inferencing_mode = False

    def get_inferencing_mode(self):
        ''' getter '''
        return self.__inferencing_mode
    
    def set_inferencing_mode(self, value):
        ''' setter '''
        self.__inferencing_mode = value
    
    inferencing_mode = property(get_inferencing_mode, set_inferencing_mode)

    def __init__(
        self,
        batch_size=25,
        map_size=(50, 50), 
        moving_max_dist=3,
        possible_n=10,
        memory_num=3,
        repeating_penalty=0.5,
        ctx=mx.gpu(),
    ):
        '''
        Init.

        Args:
            map_size:               Size of map.
            memory_num:             The number of step of agent's memory.
            repeating_penalty:      The value of penalty in the case that agent revisit.
        '''
        self.__batch_size = batch_size
        self.__map_arr = self.__create_map(map_size)
        self.__agent_pos_arr = np.array(
            [
                self.START_POS
            ] * self.__batch_size
        )
        self.__possible_n = possible_n
        self.__repeating_penalty = repeating_penalty
        self.__ctx = ctx

        self.__state_arr = self.extract_now_state()

        self.__route_memory_list = [[] for _ in range(self.__batch_size)]
        self.__memory_num = memory_num
        self.__moving_max_dist = moving_max_dist

    def reset_agent_pos(self):
        self.__agent_pos_arr = np.array(
            [
                self.START_POS
            ] * self.__batch_size
        )
        self.__state_arr = self.extract_now_state()

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

        m = m // 2
        n = n // 2

        SPACE = self.SPACE
        WALL = self.WALL
        START = self.START
        GOAL = self.GOAL

        memo = np.array([i for i in range(n * m)])
        memo = memo.reshape(m, n)

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

            if (x2 < 0) or (x2 >= m) or (y2 < 0) or (y2 >= n):
                continue

            if memo[x1, y1] != memo[x2, y2]:
                tmp_min = min(memo[x1, y1], memo[x2, y2])
                tmp_max = max(memo[x1, y1], memo[x2, y2])

                memo[memo == tmp_max] = tmp_min

                maze[x1 + x2 + 1][y1 + y2 + 1] = SPACE

        maze_arr = np.array(maze)
        return maze_arr

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        possible_action_arr = np.zeros((
            self.__state_arr.shape[0],
            self.__possible_n,
            self.__state_arr.shape[1],
            self.__state_arr.shape[2],
            self.__state_arr.shape[3],
        ))
        for batch in range(self.__batch_size):
            state_arr = self.__state_arr[batch, 0].asnumpy()
            agent_x, agent_y = np.where(state_arr == 1)
            agent_x, agent_y = agent_x[0], agent_y[0]

            if self.inferencing_mode is False:
                agent_x_arr, agent_y_arr = np.where(self.__map_arr == self.SPACE)
                key = np.random.randint(low=0, high=agent_x_arr.shape[0])
                random_agent_x = agent_x_arr[key]
                random_agent_y = agent_y_arr[key]

                state_arr[agent_x, agent_y] = 0
                state_arr[random_agent_x, random_agent_y] = 1

                self.__state_arr[batch, 0] = nd.ndarray.array(
                    state_arr,
                    ctx=self.__ctx
                )

            x_y_list = []
            for dist in range(1, self.__moving_max_dist):
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        if x == 0 and y == 0:
                            continue
                        x_y_list.append((x * dist, y * dist))

            x_y_arr = np.array(x_y_list)

            _possible_action_arr = None
            for i in range(x_y_arr.shape[0]):
                x = x_y_arr[i][0]
                y = x_y_arr[i][1]

                next_x = agent_x + x
                if next_x < 0 or next_x >= state_arr.shape[1]:
                    continue
                next_y = agent_y + y
                if next_y < 0 or next_y >= state_arr.shape[0]:
                    continue

                if self.inferencing_mode is True:
                    if (next_x, next_y) in self.__route_memory_list[batch]:
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

                    goal_flag = False
                    if x > 0:
                        for add_x in range(1, x):
                            if self.__map_arr[agent_x + add_x, next_y] == self.GOAL:
                                goal_flag = True
                    elif x < 0:
                        for add_x in range(x, 0):
                            if self.__map_arr[agent_x + add_x, next_y] == self.GOAL:
                                goal_flag = True

                    if goal_flag is False:
                        if y > 0:
                            for add_y in range(1, y):
                                if self.__map_arr[next_x, agent_y + add_y] == self.GOAL:
                                    goal_flag = True
                        elif y < 0:
                            for add_y in range(y, 0):
                                if self.__map_arr[next_x, agent_y + add_y] == self.GOAL:
                                    goal_flag = True

                    if goal_flag is True:
                        next_x = self.__goal_pos[0]
                        next_y = self.__goal_pos[1]

                next_action_arr = np.zeros((state_arr.shape[0], state_arr.shape[1]))
                next_action_arr[next_x, next_y] = 1
                next_action_arr = np.expand_dims(next_action_arr, axis=0)
                if _possible_action_arr is None:
                    _possible_action_arr = next_action_arr
                else:
                    _possible_action_arr = np.concatenate(
                        [
                            _possible_action_arr, 
                            next_action_arr
                        ],
                        axis=0
                    )

            if _possible_action_arr is None:
                for i in range(x_y_arr.shape[0]):
                    x = x_y_arr[i][0]
                    y = x_y_arr[i][1]

                    next_x = agent_x + x
                    if next_x < 0 or next_x >= state_arr.shape[1]:
                        continue
                    next_y = agent_y + y
                    if next_y < 0 or next_y >= state_arr.shape[0]:
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


                    goal_flag = False
                    if x > 0:
                        for add_x in range(1, x):
                            if self.__map_arr[agent_x + add_x, next_y] == self.GOAL:
                                goal_flag = True
                    elif x < 0:
                        for add_x in range(x, 0):
                            if self.__map_arr[agent_x + add_x, next_y] == self.GOAL:
                                goal_flag = True

                    if goal_flag is False:
                        if y > 0:
                            for add_y in range(1, y):
                                if self.__map_arr[next_x, agent_y + add_y] == self.GOAL:
                                    goal_flag = True
                        elif y < 0:
                            for add_y in range(y, 0):
                                if self.__map_arr[next_x, agent_y + add_y] == self.GOAL:
                                    goal_flag = True

                    if goal_flag is True:
                        next_x = self.__goal_pos[0]
                        next_y = self.__goal_pos[1]

                    next_action_arr = np.zeros((state_arr.shape[0], state_arr.shape[1]))
                    next_action_arr[next_x, next_y] = 1
                    next_action_arr = np.expand_dims(next_action_arr, axis=0)
                    if _possible_action_arr is None:
                        _possible_action_arr = next_action_arr
                    else:
                        _possible_action_arr = np.concatenate(
                            [
                                _possible_action_arr, 
                                next_action_arr
                            ],
                            axis=0
                        )

            if _possible_action_arr is None:
                raise ValueError("No action option found. Please lower the `memory_num`.")

            if _possible_action_arr.shape[0] < self.__possible_n:
                row_diff = self.__possible_n - _possible_action_arr.shape[0]
                while _possible_action_arr.shape[0] < self.__possible_n:
                    _possible_action_arr = np.concatenate(
                        [
                            _possible_action_arr,
                            _possible_action_arr[:row_diff]
                        ],
                        axis=0
                    )

            if _possible_action_arr.shape[0] > self.__possible_n:
                key_arr = np.arange(_possible_action_arr.shape[0])
                np.random.shuffle(key_arr)
                _possible_action_arr = _possible_action_arr[key_arr[:self.__possible_n]]

            # Forget oldest memory and do recuresive executing.
            while len(self.__route_memory_list[batch]) > self.__memory_num:
                self.__route_memory_list[batch] = self.__route_memory_list[batch][1:]

            possible_action_arr[batch, :, 0] = _possible_action_arr
            possible_action_arr[batch, :, 1] = self.__map_arr

        possible_action_arr = nd.ndarray.array(possible_action_arr, ctx=self.__ctx)
        return possible_action_arr, None

    def observe_state(self, state_arr, meta_data_arr):
        '''
        Observe states of agents in last epoch.

        Args:
            state_arr:      Tensor of state.
            meta_data_arr:  meta data of the state.
        '''
        self.__state_arr = state_arr
        self.__state_meta_data_arr = meta_data_arr

    def observe_reward_value(
        self, 
        state_arr, 
        action_arr,
        meta_data_arr=None,
    ):
        '''
        Compute the reward value.
        
        Args:
            state_arr:              Tensor of state.
            action_arr:             Tensor of action.
            meta_data_arr:          Meta data of actions.

        Returns:
            Reward value.
        '''
        reward_arr = self.__check_goal_flag(action_arr)
        for i in range(action_arr.shape[0]):
            _action_arr = action_arr[i, 0].asnumpy()
            x, y = np.where(_action_arr == 1)
            x, y = x[0], y[0]

            goal_x, goal_y = self.__goal_pos
            
            if x == goal_x and y == goal_y:
                distance = 0.0
            else:
                distance = np.sqrt(((x - goal_x) ** 2) + (y - goal_y) ** 2)

            if self.inferencing_mode is False:
                state_arr = self.__state_arr

            if state_arr is not None:
                _state_arr = state_arr[i, 0].asnumpy()
                pre_x, pre_y = np.where(_state_arr == 1)

                if pre_x == goal_x and pre_y == goal_y:
                    pre_distance = 0.0
                else:
                    pre_distance = np.sqrt(((pre_x - goal_x) ** 2) + (pre_y - goal_y) ** 2)

                distance_penalty = distance - pre_distance
                if distance_penalty == 0:
                    distance_penalty = 1
            else:
                distance_penalty = 0

            max_distance = (goal_x ** 2) + (goal_y ** 2)
            reward_arr[i] = reward_arr[i] + (max_distance - distance) - distance_penalty

        reward_arr = nd.ndarray.array(reward_arr, ctx=self.__ctx)
        reward_arr = nd.sigmoid(reward_arr / max_distance)

        return reward_arr

    def extract_now_state(self):
        '''
        Extract now map state.
        
        Returns:
            `np.ndarray` of state.
        '''
        state_arr = np.zeros(
            (
                self.__batch_size,
                2,
                self.__map_arr.shape[0],
                self.__map_arr.shape[1],
            )
        )
        for i in range(state_arr.shape[0]):
            x_y_arr = self.__agent_pos_arr[i]
            x = x_y_arr[0]
            y = x_y_arr[1]
            state_arr[i, 0, x, y] = 1
            state_arr[i, 1] = self.__map_arr
        return nd.ndarray.array(state_arr, ctx=self.__ctx)

    def update_state(
        self, 
        action_arr, 
        meta_data_arr=None
    ):
        '''
        Update state.
        
        This method can be overrided for concreate usecases.

        Args:
            action_arr:     action in `self.t`.
            meta_data_arr:  meta data of the action.
        
        Returns:
            Tuple data.
            - state in `self.t+1`.
            - meta data of the state.
        '''
        action_arr = action_arr.asnumpy()
        for i in range(action_arr.shape[0]):
            x, y = np.where(action_arr[0, 0] == 1)
            self.__agent_pos_arr[i] = np.array([x[0], y[0]])

            if self.inferencing_mode is True:
                self.__route_memory_list[i].append((x[0], y[0]))

        return self.extract_now_state(), meta_data_arr

    def __check_goal_flag(self, state_arr):
        goal_arr = np.zeros((state_arr.shape[0]))
        state_arr = state_arr.asnumpy()

        for i in range(state_arr.shape[0]):
            x, y = np.where(state_arr[i, 0] == 1)
            goal_x, goal_y = self.__goal_pos
            if x[0] == goal_x and y[0] == goal_y:
                goal_arr[i] = 1
            else:
                goal_arr[i] = 0

        return goal_arr

    def check_the_end_flag(self, state_arr, meta_data_arr=None):
        '''
        Check the end flag.

        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_arr:      state in `self.t`.
            meta_data_arr:  meta data of the state.

        Returns:
            bool
        '''
        if state_arr is None:
            return False

        goal_arr = self.__check_goal_flag(state_arr)
        if goal_arr.sum() > 0:
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
