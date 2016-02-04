#!/user/bin/env python
# -*- coding: utf-8 -*-
import copy
import random
import sys
from rl.qlearning.greedy_q_learning import GreedyQLearning


class MazeGreedyQLearning(GreedyQLearning):
    '''
    ε-greedyなQ学習により、
    迷路探索を実行する。

    迷路のマップのプリント処理については、下記のページを参考にさせていただいた。
    http://d.hatena.ne.jp/Kshi_Kshi/20111227/1324993576

    Attributes:
        epsilon_greedy_rate:    ε-グリーディの確率の具象プロパティ。

    '''

    # ε-グリーディの確率
    __epsilon_greedy_rate = 0.5

    def get_epsilon_greedy_rate(self):
        '''
        getter
        ε-グリーディの確率
        '''
        if isinstance(self.__epsilon_greedy_rate, float) is False:
            raise TypeError("The type of __epsilon_greedy_rate must be floot.")
        return self.__epsilon_greedy_rate

    def set_epsilon_greedy_rate(self, value):
        '''
        setter
        ε-グリーディの確率
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __epsilon_greedy_rate must be float.")
        self.__epsilon_greedy_rate = value

    # 具象プロパティ
    epsilon_greedy_rate = property(get_epsilon_greedy_rate, set_epsilon_greedy_rate)

    def initialize(self, square_map_data):
        '''
        迷路を初期化する
        文字列の迷路マップデータ：square_map_dataを2次元のリスト：map_data_listに格納する。

        Args:
            square_map_data:      n×nの二次元リストに格納されることを前提としたCSV形式の文字列

        '''
        self.__map_data_list = []
        [self.__map_data_list.append(line.split(",")) for line in square_map_data.split("\n") if line.strip() != ""]
        self.__start_point = self.__decide_start_point(self.__map_data_list)
        # マップ状況をデバッグメッセージに登録
        self.__debug_map_data_list()

        for y, line in enumerate(self.__map_data_list):
            for x, v in enumerate(line):
                if v == "#":
                    continue
                point = (x, y)
                reward_tuple = self.__extract_reward_value(point)
                self.save_r_dict(point, reward_tuple[0])

    def learn(self, state_key=None, limit=1000):
        '''
        Q学習
        具象メソッド。
        '''
        if state_key is None:
            state_key = self.__decide_start_point(self.__map_data_list)

        self.t = 1
        while self.t <= limit:
            next_action_list = self.extract_possible_actions(state_key)
            action_key = self.select_action(
                state_key=state_key,
                next_action_list=next_action_list
            )
            reward_value, end_flag = self.__extract_reward_value(state_key)

            next_next_action_list = self.extract_possible_actions(action_key)

            # 現在わかっている限りの次の行動選択時の最大Q値
            next_action_key = self.predict_next_action(action_key, next_next_action_list)
            next_max_q = self.extract_q_dict(action_key, next_action_key)

            # マップ状況をデバッグメッセージに登録
            self.__debug_map_data_list(state_key, action_key, self.t, end_flag)

            # ゴールに到達すればこれ以上繰り返しは不要。
            if end_flag is True:
                break

            # 学習と更新
            self.update_q(
                state_key=state_key,
                action_key=action_key,
                reward_value=reward_value,
                next_max_q=next_max_q
            )

            # 出来事回数をカウント
            self.t += 1
            # 迷路探索の場合、tの行動がt+1の状態になる。
            state_key = action_key

    def select_action(self, state_key, next_action_list):
        '''
        状態に紐付けて行動を選択する。
        具象メソッド

        Args:
            state_key:      状態

        Retruns:
            行動

        '''
        true_rate_list = [True] * int(self.epsilon_greedy_rate * 100)
        false_rate_list = [False] * int((1 - self.epsilon_greedy_rate) * 100)

        univerce_list = []
        [univerce_list.append(true_rate) for true_rate in true_rate_list]
        [univerce_list.append(false_rate) for false_rate in false_rate_list]

        epsilon_greedy_flag = random.choice(univerce_list)

        if epsilon_greedy_flag is True:
            self.debug_message_list.append("\nGreedy Mode: On\n")
            return super().select_action(state_key, next_action_list)
        else:
            self.debug_message_list.append("\nGreedy Mode: Off\n")
            return random.choice(next_action_list)

    def extract_possible_actions(self, state_key):
        '''
        次回のs+1の状態で選択可能な行動のリストを得る。

        具象メソッド。
        引数で指定した座標から移動できる座標リストを獲得する。

        Args:
            state_key       s+1の時の座標

        Returns:
            座標から移動可能な座標リスト

        Exceptions:
            ValueError:     壁を指定している場合に発生する例外。

        '''
        x, y = state_key
        if self.__map_data_list[y][x] == "#":
            raise ValueError("It is the wall. (x, y)=(%d, %d)" % (x, y))
        around_map = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        possible_actoins_list = [(_x, _y) for _x, _y in around_map if self.__map_data_list[_y][_x] != "#" and self.__map_data_list[_y][_x] != "S"]
        return possible_actoins_list

    def __decide_start_point(self, map_data_list):
        '''
        マップデータ内のStart地点:Sの座標を返す。

        Args:
            map_data_list:      迷路のマップデータ。n×nの二次元リスト。

        Returns:
            マップデータ内のStart地点:Sの座標：(x, y)のtuple

        Exceptions:
            ValueError:         迷路のマップデータにスタート地点：Sが無い場合に発生する例外。

        '''
        for y, line in enumerate(map_data_list):
            for x, v in enumerate(line):
                if v == "S":
                    return (x, y)

        raise ValueError("Fieldにスタート地点：Sがありません。")

    def __extract_reward_value(self, point):
        '''
        指定した座標のfieldの値を返す. エピソード終了判定もする。

        Args:
            point:      現在の位置座標

        Returns:
            (報酬, 終了判定)のtuple

        Exceptions:
            ValueError:   エージェントが壁にぶつかった場合に発生する例外。

        '''
        x, y = point

        if self.__map_data_list[y][x] == "G":
            return 100.0, True
        elif self.__map_data_list[y][x] == "S":
            return 0.0, False
        elif self.__map_data_list[y][x] == "#":
            raise ValueError("It is the wall. (x, y)=(%d, %d)" % (x, y))
        else:
            v = float(self.__map_data_list[y][x])
            return v, False

    def __debug_map_data_list(self, state_key=None, action_key=None, i=0, end_flag=False):
        '''
        マップ情報や現在位置の情報をデバッグメッセージに追加する。

        Args:
            state_key:      現在位置(x, y)のtuple
            action_key:     次回行動時の位置(x, y)のtuple

        '''
        map_data_list = copy.deepcopy(self.__map_data_list)
        if state_key is not None:
            x, y = state_key
            map_data_list[y][x] = "@"
            point = state_key
        else:
            point = ""

        self.debug_message_list.append("\n\t----- Map Field: %s -----" % str(point))
        for line in map_data_list:
            self.debug_message_list.append("\t" + "%3s " * len(line) % tuple(line))

        if state_key is not None and action_key is not None:
            self.debug_message_list.append(
                "\n\tMove count:" + str(i) + "\n"
            )
            self.debug_message_list.append(
                "\tState: %s -> Action:%s\n" % (state_key, action_key)
            )
            self.debug_message_list.append(
                "\tQ-Value: %s" % (str(self.extract_q_dict(state_key, action_key))) +
                "\tReward: %s\n" % (str(self.extract_r_dict(state_key)))
            )

        if i == 0:
            self.debug_message_list.append("\n\tMission Start !!\n")

        if end_flag is True:
            self.debug_message_list.append("\nMission complete !!\n")

    def __del__(self):
        '''
        デストラクタ
        '''
        if self.debug_mode is True:
            self.debug_message_list.append("\nUpdate Count:" + str(self.t))

        super().__del__()

if __name__ == "__main__":
    # "S": Start地点, "#": 壁, "数値": 報酬
    RAW_Field = """
#,#,#,#,#,#,#
#,S,1,1,-10,1,#
#,10,-10,1,1,1,#
#,10,-10,1,-10,1,#
#,20,20,30,-10,1,#
#,1,-10,10,100,G,#
#,#,#,#,#,#,#
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

    greedy_rate = 0.3
    if len(sys.argv) > 5:
        greedy_rate = float(sys.argv[5])

    maze_q_learning = MazeGreedyQLearning()
    maze_q_learning.epsilon_greedy_rate = greedy_rate
    maze_q_learning.debug_mode = debug_mode
    maze_q_learning.alpha_value = alpha_value
    maze_q_learning.gamma_value = gamma_value
    maze_q_learning.initialize(RAW_Field)
    maze_q_learning.learn(limit=limit)
