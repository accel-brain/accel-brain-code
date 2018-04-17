# -*- coding: utf-8 -*-
import sys
import numpy as np
from devsample.maze_boltzmann_q_learning import MazeBoltzmannQLearning


if __name__ == "__main__":
    # "S": Start point, "G": End point(goal), "#": wall, "#": Agent.
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

    maze_q_learning = MazeBoltzmannQLearning()
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
