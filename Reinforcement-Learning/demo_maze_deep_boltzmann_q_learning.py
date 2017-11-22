import sys
import numpy as np
from devsample.maze_greedy_q_learning import MazeGreedyQLearning
from pydbm.dbm.deepboltzmannmachine.stacked_auto_encoder import StackedAutoEncoder
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.activation.logistic_function import LogisticFunction


if __name__ == "__main__":
    # "S": Start point, "G": End point(goal), "#": wall, "#": Agent.
    start_point_label, end_point_label, wall_label, agent_label = ("S", "G", "#", "@")
    map_d = 10
    map_arr = np.random.rand(map_d, map_d)
    map_arr += np.diag(list(range(map_d)))

    vector_list_list = []
    for y in range(map_arr.shape[0]):
        for x in range(map_arr.shape[1]):
            vector_list = []
            vector_list.append(map_arr[y][x])
            for _x in [-1, 1]:
                __x = x + _x
                for _y in [-1, 1]:
                    __y = y + _y
                    if __x < 0 or __y < 0:
                        vector = 0
                    else:
                        try:
                            vector = map_arr[__y][__x]
                        except IndexError:
                            vector = 0
                    vector_list.append(vector)
            vector_list_list.append(vector_list)
    vector_arr = np.array(vector_list_list)
    vector_arr = vector_arr.astype(float)
    dbm = StackedAutoEncoder(
        DBMMultiLayerBuilder(),
        [vector_arr.shape[1], vector_arr.shape[1], 10],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.005
    )
    dbm.learn(vector_arr, traning_count=1)
    feature_arr = dbm.feature_points_arr
    feature_arr = feature_arr[:, 0]
    feature_map_arr = feature_arr.reshape(map_d, map_d)

    map_arr = map_arr.astype(object)
    map_arr[:, 0] = wall_label
    map_arr[0, :] = wall_label
    map_arr[:, -1] = wall_label
    map_arr[-1, :] = wall_label
    map_arr[1][1] = start_point_label
    map_arr[map_d - 2][map_d - 2] = end_point_label

    feature_map_arr = feature_map_arr.astype(object)
    feature_map_arr[:, 0] = wall_label
    feature_map_arr[0, :] = wall_label
    feature_map_arr[:, -1] = wall_label
    feature_map_arr[-1, :] = wall_label
    feature_map_arr[1][1] = start_point_label
    feature_map_arr[map_d - 2][map_d - 2] = end_point_label

    limit = 10000
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    alpha_value = 0.9
    if len(sys.argv) > 2:
        alpha_value = float(sys.argv[2])

    gamma_value = 0.9
    if len(sys.argv) > 3:
        gamma_value = float(sys.argv[3])

    greedy_rate = 0.25
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

    maze_deep_boltzmann_q_learning = MazeGreedyQLearning()
    maze_deep_boltzmann_q_learning.epsilon_greedy_rate = greedy_rate
    maze_deep_boltzmann_q_learning.alpha_value = alpha_value
    maze_deep_boltzmann_q_learning.gamma_value = gamma_value
    maze_deep_boltzmann_q_learning.initialize(
        map_arr=feature_map_arr,
        start_point_label=start_point_label,
        end_point_label=end_point_label,
        wall_label=wall_label,
        agent_label=agent_label
    )
    maze_deep_boltzmann_q_learning.learn(state_key=(1, 1), limit=limit)

    print("The number of learning (not deep boltzmann machine): " + str(maze_q_learning.t))
    print("The number of learning (deep boltzmann machine): " + str(maze_deep_boltzmann_q_learning.t))
