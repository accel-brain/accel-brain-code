#!/user/bin/env python
# -*- coding: utf-8 -*-
import sys
from rl.qlearning.interface.maze_q_learning_interface import MazeQLearningInterface
from deeplearning.dbm.deep_boltzmann_machine import DeepBoltzmannMachine


class MazeDeepBoltzmannQLearning(object):
    '''
    深層強化学習による迷路探索

    深層ボルツマンマシンを自己符号化器のように利用することで
    状態の特徴を周辺の状態との関連から事前学習した上で
    Q学習を実行する
    '''

    # 迷路探索のインターフェイス
    __maze_q_learning_interface = None

    def get_maze_q_learning_interface(self):
        if isinstance(self.__maze_q_learning_interface, MazeQLearningInterface) is False:
            raise TypeError()
        return self.__maze_q_learning_interface

    def set_maze_q_learning_interface(self, value):
        if isinstance(value, MazeQLearningInterface) is False:
            raise TypeError()
        self.__maze_q_learning_interface = value

    # 迷路探索のインターフェイスを実現したオブジェクトのプロパティ
    maze_q_learning_interface = property(get_maze_q_learning_interface, set_maze_q_learning_interface)

    # 深層ボルツマンマシンのオブジェクト
    __dbm = None

    # 訓練回数
    __traning_count = 2

    # 各地点の報酬
    __point_reward = {
        "S": 1,    # スタート地点
        "G": 100,  # ゴール地点
        "#": 0,  # 壁
        "NULL": 0  # scope_limit分進んだ場合にマップ外になる場合
    }

    # 探索範囲
    __scope_limit = 1

    def __init__(
        self,
        point_reward=None,
        scope_limit=1,
        traning_count=2,
        learning_rate=0.1
    ):
        '''
        初期化する

        Args:
            point_reward:         壁、スタート地点、ゴール地点の報酬
            scope_limit:          深層ボルツマンマシンによる探索範囲
                                  （例） 距離的な概念であるため
                                       1なら上下左右斜め1個分隣のマスが範囲になる
            traning_count:        深層ボルツマンマシンの訓練回数
            learning_rate:        深層ボルツマンマシンの学習率

        '''
        if point_reward is not None:
            self.__point_reward = point_reward
        self.__scope_limit = scope_limit

        neuron_count = self.__decide_neuron_count(scope_limit)
        self.__dbm = DeepBoltzmannMachine(
            DBMMultiLayerBuilder(),
            [neuron_count, 3, 1],
            SigmoidFunction(),
            ContrastiveDivergence(),
            learning_rate
        )
        self.__traning_count = traning_count

    def initialize(self, square_map_data):
        '''
        迷路を初期化する
        文字列の迷路マップデータ：square_map_dataを2次元のリスト：map_data_listに格納する。
        深層ボルツマンマシンにより、迷路マップデータの特徴を学習した上でQ学習を開始するべく、
        前処理を実行していく

        Args:
            square_map_data:      n×nの二次元リストに格納されることを前提としたCSV形式の文字列
        '''
        map_data_matrix = []
        [map_data_matrix.append(line.split(",")) for line in square_map_data.split("\n") if line.strip() != ""]

        all_count = len(map_data_matrix) * len(map_data_matrix[0])
        map_vector_matrix = []
        vector_matrix = []
        for i in range(len(map_data_matrix)):
            map_vector_list = []
            for j in range(len(map_data_matrix[i])):
                vector_list = []
                for k in range(i - self.__scope_limit, i + self.__scope_limit + 1):
                    for l in range(j - self.__scope_limit, j + self.__scope_limit + 1):
                        if k < 0 or k >= len(map_data_matrix):
                            vector_list.append(self.__point_reward["NULL"])
                        else:
                            if l < 0 or l >= len(map_data_matrix[k]):
                                vector_list.append(self.__point_reward["NULL"])
                            else:
                                if map_data_matrix[k][l] in self.__point_reward:
                                    reward = self.__point_reward[map_data_matrix[k][l]] / all_count
                                else:
                                    reward = float(map_data_matrix[k][l]) / all_count
                                vector_list.append(reward)

                map_vector_list.append(vector_list)
                vector_matrix.append(vector_list)
            map_vector_matrix.append(map_vector_list)

        self.__dbm.learn(vector_matrix, traning_count=self.__traning_count)

        map_feature_matrix = []
        for i in range(len(map_vector_matrix)):
            map_feature_list = []
            for j in range(len(map_vector_matrix[i])):
                # ヘブ則的連想
                self.__dbm.learn([map_vector_matrix[i][j]], traning_count=1)
                if map_data_matrix[i][j] not in self.__point_reward:
                    map_feature_list.append(
                        self.__dbm.get_feature_point_list(1)[0]
                    )
                else:
                    map_feature_list.append(
                        map_data_matrix[i][j]
                    )

            map_feature_matrix.append(map_feature_list)

        square_map_data = "\n".join(
            [",".join([str(map_feature) for map_feature in map_feature_list]) for map_feature_list in map_feature_matrix]
        )

        self.maze_q_learning_interface.initialize(square_map_data)

    def learning(self, limit=1000):
        '''
        Q学習を実行する

        Args:
            limit:      学習回数
        '''
        self.maze_q_learning_interface.learn(limit=limit)

    def __decide_neuron_count(self, scope_limit):
        '''
        探索範囲に応じてニューロン数を調節する

        Args:
            scope_limit:    探索範囲
        '''
        return ((scope_limit * 2) + 1) ** 2

if __name__ == "__main__":
    from deeplearning.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
    from deeplearning.approximation.contrastive_divergence import ContrastiveDivergence
    from nlp.textmining.extractbigrams.save_positive_bigrams import SavePositiveBigrams
    from deeplearning.activation.sigmoid_function import SigmoidFunction
    from rl.qlearning.boltzmannqlearning.maze_boltzmann_q_learning import MazeBoltzmannQLearning

    # "S": Start地点, "#": 壁, "数値": 報酬
    RAW_Field = """
#,#,#,#,#,#,#,#,#,#,#,#
#,S,10,10,0,1,0,1,10,5,1,#
#,0,0,1,10,1,0,0,5,1,10,#
#,10,0,1,0,30,10,3,1,3,20,#
#,10,20,30,0,1,10,2,0,0,1,#
#,1,0,10,100,G,1,1,1,0,3,#
#,#,#,#,#,#,#,#,#,#,#,#
"""

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        debug_mode = True
    else:
        debug_mode = False

    limit = 1000
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])

    alpha_value = 0.9
    if len(sys.argv) > 3:
        alpha_value = float(sys.argv[3])

    gamma_value = 0.9
    if len(sys.argv) > 4:
        gamma_value = float(sys.argv[4])

    time_rate = 0.001
    if len(sys.argv) > 5:
        time_rate = float(sys.argv[5])

    maze_q_learning = MazeBoltzmannQLearning()
    maze_q_learning.time_rate = time_rate
    maze_q_learning.debug_mode = debug_mode
    maze_q_learning.alpha_value = alpha_value
    maze_q_learning.gamma_value = gamma_value

    maze_deep_boltzmann_q_learning = MazeDeepBoltzmannQLearning(scope_limit=2)
    maze_deep_boltzmann_q_learning.maze_q_learning_interface = maze_q_learning
    maze_deep_boltzmann_q_learning.initialize(RAW_Field)
    maze_deep_boltzmann_q_learning.learning(limit)
