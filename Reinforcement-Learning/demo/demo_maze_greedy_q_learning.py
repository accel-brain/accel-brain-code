# -*- coding: utf-8 -*-
import sys
import random
import numpy as np
import copy
from pyqlearning.qlearning.greedy_q_learning import GreedyQLearning


class MazeGreedyQLearning(GreedyQLearning):
    '''
    ε-greedy Q-Learning to solve Maze problem.

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
                    
                self.save_r_df((x, y), float(arr_value))

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
            self.save_r_df(state_key, reward_value)
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

    def inference(self, limit=1000):
        '''
        Inference route.
        
        Args:
            limit:    the number of inferencing.
        
        Returns:
            [(x_1, y_1), (x_2, y_2), ...]
        '''
        route_list = []
        memory_list = []
        state_key = self.__start_point_tuple
        x, y = state_key
        end_x, end_y = self.__end_point_tuple
        for i in range(limit):
            q_df = self.q_df[self.q_df.state_key == state_key]
            if len(memory_list):
                q_df = q_df[~q_df.action_key.isin(memory_list)]
            if q_df.shape[0] > 1:
                q_df = q_df.sort_values(by=["q_value"], ascending=False)
                action_key = q_df.iloc[0, :]["action_key"]
                q_value = q_df.iloc[0, :]["q_value"]
            elif q_df.shape[0] == 1:
                action_key = q_df.action_key.values[0]
                q_value = q_df.q_value.values[0]
            else:
                action_key_list = self.extract_possible_actions(state_key)
                action_key_list = [v for v in action_key_list if v not in memory_list]
                q_value = 0.0
                if len(action_key_list):
                    action_key = random.choice(action_key_list)
                    _q_df = q_df[q_df.action_key == action_key]
                    if _q_df.shape[0]:
                        q_value = _q_df.q_value.values[0]

            state_key = self.update_state(
                state_key=state_key,
                action_key=action_key
            )
            x, y = state_key
            route_list.append((x, y, q_value))
            memory_list.append(state_key)
            if self.check_the_end_flag(state_key) is True:
                break

        return route_list


if __name__ == "__main__":
    # "S": Start point, "G": End point(goal), "#": wall, "@": Agent.
    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_d = 10
    map_arr = 10 * np.random.rand(map_d, map_d)
    map_arr = map_arr.astype(int)
    map_arr += np.diag(list(range(map_d))) * 10
    map_arr = map_arr.astype(object)
    map_arr[:, 0] = wall_label
    map_arr[0, :] = wall_label
    map_arr[:, -1] = wall_label
    map_arr[-1, :] = wall_label
    map_arr[1][1] = start_point_label
    map_arr[map_d - 2][map_d - 2] = end_point_label

    limit = 10000
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    alpha_value = 0.9
    if len(sys.argv) > 2:
        alpha_value = float(sys.argv[2])

    gamma_value = 0.9
    if len(sys.argv) > 3:
        gamma_value = float(sys.argv[3])

    greedy_rate = 0.75
    if len(sys.argv) > 4:
        greedy_rate = float(sys.argv[4])

    maze_q_learning = MazeGreedyQLearning()
    maze_q_learning.epsilon_greedy_rate = greedy_rate
    maze_q_learning.alpha_value = alpha_value
    maze_q_learning.gamma_value = gamma_value
    maze_q_learning.initialize(
        map_arr=map_arr,
        start_point_label=start_point_label,
        end_point_label=end_point_label,
        wall_label=wall_label,
        agent_label=agent_label
    )
    maze_q_learning.learn(state_key=(1, 1), limit=limit)
    
    q_df = maze_q_learning.q_df
    q_df = q_df.sort_values(by=["q_value"], ascending=False)
    print(q_df.head())
