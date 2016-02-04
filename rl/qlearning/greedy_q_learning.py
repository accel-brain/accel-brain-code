#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractproperty
from rl.q_learning import QLearning


class GreedyQLearning(QLearning):
    '''
    ε-greedyなQ学習を実行する。

    Attributes:
        epsilon_greedy_rate:    ε-グリーディの確率の抽象プロパティ

    '''

    @abstractproperty
    def epsilon_greedy_rate(self):
        '''
        抽象プロパティ
        ε-グリーディの確率
        '''
        raise NotImplementedError("This property must be implemented.")

    # ε-グリーディで貪欲に探索した回数
    __epsilon_greedy_count = 0

    def select_action(self, state_key, next_action_list):
        '''
        状態に紐付けて行動を選択する。
        具象クラス
        ε-greedyに、Q値が最大になる場合の探索を行なう。

        Args:
            state_key:      状態

        Retruns:
            (行動, 最大報酬値)のtuple

        '''

        action_key = self.predict_next_action(state_key, next_action_list)
        self.__epsilon_greedy_count += 1

        # デバッグメッセージを追加
        self.__debug_select_action(state_key, action_key)

        return action_key

    def __debug_select_action(self, state_key, action_key):
        '''
        デバッグメッセージを追加する

        Args:
            state_key:      状態
            action_key:     行動
        '''
        if self.debug_mode is True:
            max_q = self.extract_q_dict(state_key, action_key)
            self.debug_message_list.append("\nGreedy Mode: On\n")
            self.debug_message_list.append(
                "Greedy ..."
            )
            self.debug_message_list.append(
                "Max Q: " + str(max_q) + "\tNext Action: " + str(action_key)
            )

    def __del__(self):
        '''
        デストラクタ
        '''
        if self.debug_mode is True:
            self.debug_message_list.append("Greedy searching: " + str(self.__epsilon_greedy_count) + "\n\n")

        super().__del__()
