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
    map_d = 20
    map_arr = 10 * np.random.rand(map_d, map_d)
    map_arr = map_arr.astype(int)
    map_arr += np.diag(list(range(map_d))) * 10

    dbm = StackedAutoEncoder(
        DBMMultiLayerBuilder(),
        [map_arr.shape[1], map_arr.shape[1], map_arr.shape[1]],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_x, traning_count=1)
    feature_map_arr = pd.DataFrame(dbm.feature_points_arr)

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
