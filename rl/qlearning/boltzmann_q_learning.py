#!/user/bin/env python
# -*- coding: utf-8 -*-
import random
import math
from rl.q_learning import QLearning


class BoltzmannQLearning(QLearning):
    '''
    Boltzmann選択によるQ学習を実行する。

    '''

    # 時間レート
    __time_rate = 0.001

    def get_time_rate(self):
        '''
        getter
        時間レート
        '''
        if isinstance(self.__time_rate, float) is False:
            raise TypeError("The type of __time_rate must be float.")

        if self.__time_rate <= 0.0:
            raise ValueError("The value of __time_rate must be greater than 0.0")

        return self.__time_rate

    def set_time_rate(self, value):
        '''
        setter
        時間レート
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __time_rate must be float.")

        if value <= 0.0:
            raise ValueError("The value of __time_rate must be greater than 0.0")

        self.__time_rate = value

    time_rate = property(get_time_rate, set_time_rate)

    def select_action(self, state_key, next_action_list):
        '''
        状態に紐付けて行動を選択する。
        具象クラス
        ボルツマン分布で確率論的に行動を探索する。

        Args:
            state_key:      状態

        Retruns:
            (行動, 最大報酬値)のtuple

        '''
        next_action_b_list = self.__calculate_boltzmann_factor(state_key, next_action_list)

        if len(next_action_b_list) == 1:
            return next_action_b_list[0][0]

        next_action_b_list = sorted(next_action_b_list, key=lambda x: x[1])

        prob = random.random()
        i = 0
        while prob > next_action_b_list[i][1] + next_action_b_list[i + 1][1]:
            i += 1
            if i + 1 >= len(next_action_b_list):
                break

        max_b_action_key = next_action_b_list[i][0]

        # デバッグメッセージを追加
        self.__debug_select_action(state_key, max_b_action_key)
        return max_b_action_key

    def __calculate_sigmoid(self):
        '''
        ボルツマンの温度関数

        Returns:
            時間：self.tとself.time_rateと共に0に収束する値

        '''
        sigmoid = 1 / math.log(self.t * self.time_rate + 1.1)
        return sigmoid

    def __calculate_boltzmann_factor(self, state_key, next_action_list):
        '''
        ボルツマン因子を計算する

        Args:
            state_key:            状態
            next_action_list:     選択可能な行動リスト

        Returns:
            当該状態で選択可能な行動のボルツマン因子のリスト
            リストの要素は(行動, ボルツマン確率)のtuple
        '''
        sigmoid = self.__calculate_sigmoid()
        parent_list = [(action_key, math.exp(self.extract_q_dict(state_key, action_key) / sigmoid)) for action_key in next_action_list]
        parent_b_list = [parent[1] for parent in parent_list]
        next_action_b_list = [(action_key, child_b / sum(parent_b_list)) for action_key, child_b in parent_list]
        return next_action_b_list

    def __debug_select_action(self, state_key, action_key):
        '''
        デバッグメッセージを追加する

        Args:
            state_key:      状態
            action_key:     行動
        '''
        if self.debug_mode is True:
            max_q = self.extract_q_dict(state_key, action_key)
            self.debug_message_list.append(
                "\nBoltzmann Selection ..."
            )
            self.debug_message_list.append(
                "Max Q: " + str(max_q) + "\tNext Action: " + str(action_key)
            )

    def __del__(self):
        '''
        デストラクタ
        '''
        if self.debug_mode is True:
            self.debug_message_list.append("Boltmann searching: " + str(self.t) + "\n\n")

        super().__del__()
