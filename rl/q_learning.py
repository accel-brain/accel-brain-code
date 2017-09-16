#!/user/bin/env python
# -*- coding: utf-8 -*-
import random
from abc import ABCMeta, abstractmethod
from operator import itemgetter


class QLearning(metaclass=ABCMeta):
    '''
    Q学習の抽象基底クラス
    構造は無難にTemplate Method Patternを選択。
    getter/setterでアルゴリズムのチューニングを行なう。

    Attributes:
        alpha_value:        学習率：アルファ値。比例して学習時の変異性が高まる。
        gamma_value:        割引率：ガンマ値。比例して行動時に獲得し得る報酬の可能性からの影響度が高まる。
        q_dict:             状態：sにおける行動：aのQ値。
        r_dict:             状態：sにおける報酬。
        t:                  時点
        debug_mode:         True:デバッグモード,  False:非デバッグモード
        debug_message_list: デバッグメッセージのリスト

    '''

    # 学習率：アルファ値。比例して学習時の変異性が高まる。
    __alpha_value = 0.1

    def get_alpha_value(self):
        '''
        getter
        学習率
        '''
        if isinstance(self.__alpha_value, float) is False:
            raise TypeError("The type of __alpha_value must be float.")
        return self.__alpha_value

    def set_alpha_value(self, value):
        '''
        setter
        学習率
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __alpha_value must be float.")
        self.__alpha_value = value

    alpha_value = property(get_alpha_value, set_alpha_value)

    # 割引率:ガンマ値。比例して、次回行動時に獲得し得る報酬からの影響度が高まる。
    __gamma_value = 0.5

    def get_gamma_value(self):
        '''
        getter
        割引率
        '''
        if isinstance(self.__gamma_value, float) is False:
            raise TypeError("The type of __gamma_value must be float.")
        return self.__gamma_value

    def set_gamma_value(self, value):
        '''
        setter
        割引率
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __gamma_value must be float.")
        self.__gamma_value = value

    gamma_value = property(get_gamma_value, set_gamma_value)

    # Q値
    __q_dict = {}

    def get_q_dict(self):
        '''
        getter
        Q値
        '''
        if isinstance(self.__q_dict, dict) is False:
            raise TypeError("The type of __q_dict must be dict.")
        return self.__q_dict

    def set_q_dict(self, value):
        '''
        setter
        Q値
        '''
        if isinstance(value, dict) is False:
            raise TypeError("The type of __q_dict must be dict.")
        self.__q_dict = value

    q_dict = property(get_q_dict, set_q_dict)

    def extract_q_dict(self, state_key, action_key):
        '''
        プロパティ：q_dictからキーを指定して取得する
        キーが存在しない場合はデフォルト値を返す

        Args:
            state_key:      状態
            action_key:     行動

        Returns:
            Q値

        '''
        q = 0.0
        try:
            q = self.q_dict[state_key][action_key]
        except KeyError:
            self.save_q_dict(state_key, action_key, q)

        return q

    def save_q_dict(self, state_key, action_key, q_value):
        '''
        プロパティ：q_dictにキーを指定して保存する
        キーが存在しない場合はデフォルト値を返す

        Args:
            state_key:      状態
            action_key:     行動
            q_value:        Q値

        Exceptions:
            TypeError:      q_valueがfloat型ではない場合に発生する例外

        '''
        if isinstance(q_value, float) is False:
            raise TypeError("The type of q_value must be float.")

        if state_key not in self.q_dict:
            self.q_dict[state_key] = {action_key: q_value}
        else:
            self.q_dict[state_key][action_key] = q_value

    # 状態ごとの報酬
    __r_dict = {}

    def get_r_dict(self):
        '''
        getter
        状態ごとの報酬
        '''
        if isinstance(self.__r_dict, dict) is False:
            raise TypeError("The type of __r_dict must be dict.")
        return self.__r_dict

    def set_r_dict(self, value):
        '''
        setter
        状態ごとの報酬
        '''
        if isinstance(value, dict) is False:
            raise TypeError("The type of __r_dict must be dict.")
        self.__r_dict = value

    r_dict = property(get_r_dict, set_r_dict)

    def extract_r_dict(self, state_key, action_key=None):
        '''
        プロパティ：r_dictからキーを指定して取得する

        Args:
            state_key:     状態
            action_key:    行動

        Returns:
            報酬

        '''
        try:
            if action_key is None:
                return self.r_dict[state_key]
            else:
                return self.r_dict[(state_key, action_key)]
        except KeyError:
            self.save_r_dict(state_key, 0.0, action_key)
            return self.extract_r_dict(state_key, action_key)

    def save_r_dict(self, state_key, r_value, action_key=None):
        '''
        プロパティ：r_dictにキーを指定して保存する

        Args:
            state_key:     状態
            r_value:       報酬
            action_key:    行動

        '''
        if isinstance(r_value, float) is False:
            raise TypeError("The type of r_value must be float.")

        if action_key is not None:
            self.r_dict[(state_key, action_key)] = r_value
        else:
            self.r_dict[state_key] = r_value

    # 状態と行動の時間
    __t = 0

    def get_t(self):
        '''
        getter
        状態と行動の時間
        '''
        if isinstance(self.__t, int) is False:
            raise TypeError("The type of __t must be int.")
        return self.__t

    def set_t(self, value):
        '''
        setter
        状態と行動の時間
        '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __t must be int.")
        self.__t = value

    t = property(get_t, set_t)

    # デバッグモード
    __debug_mode = False

    def get_debug_mode(self):
        if isinstance(self.__debug_mode, bool) is False:
            raise TypeError("The type of __debug_mode must be bool.")
        return self.__debug_mode

    def set_debug_mode(self, value):
        if isinstance(value, bool) is False:
            raise TypeError("The type of __debug_mode must be bool.")
        self.__debug_mode = value

    debug_mode = property(get_debug_mode, set_debug_mode)

    # デバッグメッセージ
    __debug_message_list = []

    def get_debug_message_list(self):
        if isinstance(self.__debug_message_list, list) is False:
            raise TypeError("The type of __debug_message_list must be list.")
        return self.__debug_message_list

    def set_debug_message_list(self, value):
        if self.debug_mode is True:
            if isinstance(value, list) is False:
                raise TypeError("The type of __debug_message_list must be list.")
            self.__debug_message_list = value

    debug_message_list = property(get_debug_message_list, set_debug_message_list)

    @abstractmethod
    def learn(self, state_key, limit):
        '''
        Q学習を実行する。
        迷路探索やバンディットアルゴリズムへの応用など、
        具象的なユースケースが複数想定されるため、抽象メソッドに留めておく。

        Args:
            state_key:      状態
            limit:          学習回数

        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def select_action(self, state_key, next_action_list):
        '''
        状態に紐付けて行動を選択する。
        迷路探索やバンディットアルゴリズムへの応用など、
        具象的なユースケースが複数想定されるため、抽象メソッドに留めておく。

        Args:
            state_key:              状態
            next_action_list:       t+1の次回行動時に可能な行動。空なら全ての行動が可能と見做す。

        Retruns:
            (行動, Qの最大値)のtuple

        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def extract_possible_actions(self, state_key):
        '''
        次回のs+1の状態で選択可能な行動のリストを得る。

        抽象メソッド。
        状態を前提とした行動選択は迷路探索などの具象的な環境で異なるため、
        下位クラスに実現を委ねる。

        Args:
            state_key       s+1の状態

        Returns:
            選択可能な行動リスト

        '''
        raise NotImplementedError("This method must be implemented.")

    def update_q(self, state_key, action_key, reward_value, next_max_q):
        '''
        Q値を更新する

        Args:
            state_key:              状態
            action_key:             行動
            reward_value:           状態：state_keyにおける報酬
            next_max_q:             次回行動時の最大Q値

        '''
        # 現在のQ値
        q = self.extract_q_dict(state_key, action_key)
        # Q値を更新する
        new_q = q + self.alpha_value * (reward_value + (self.gamma_value * next_max_q) - q)
        # デバッグメッセージを追加する
        self.__debug_update_q(state_key, action_key, reward_value, next_max_q, new_q)
        # 更新後のQ値を登録する
        self.save_q_dict(state_key, action_key, new_q)

    def predict_next_action(self, state_key, next_action_list):
        '''
        Q値が最大になる次回の行動を予測する。

        Args:
            state_key:          状態(S_t+1)
            next_action_list:   t+1の状態で採り得る行動リスト

        Returns:
            行動

        '''
        next_action_q_list = [(action_key, self.extract_q_dict(state_key, action_key)) for action_key in next_action_list]
        # 最大値が複数個ある場合、若いキーが一律に選択されてしまうため
        random.shuffle(next_action_q_list)
        max_q_action = max(next_action_q_list, key=itemgetter(1))[0]

        return max_q_action

    def __debug_update_q(self, state_key, action_key, reward_value, next_max_q, new_q):
        '''
        Q値を更新した際のデバッグメッセージを追加する

        Args:
            state_key:              状態
            action_key:             行動
            reward_value:           状態：state_keyにおける報酬
            next_max_q:             次回行動時の最大Q値
            new_q:                  更新後のQ値

        '''
        q = self.extract_q_dict(state_key, action_key)

        self.debug_message_list.append(
            "\tQ(S_t, A_t) = " + str(q)
        )
        self.debug_message_list.append(
            "\tα-value = " + str(self.alpha_value)
        )
        self.debug_message_list.append(
            "\tγ-value = " + str(self.gamma_value)
        )
        self.debug_message_list.append(
            "\tpossible max Q = " + str(next_max_q)
        )
        self.debug_message_list.append(
            "\tReward value = " + str(reward_value)
        )
        self.debug_message_list.append(
            "\tQ(S_t+1, A_t+1) = Q(S_t, A_t) + α-value * (Reward value + γ-value * possible max Q - Q(S_t, A_t))"
        )
        self.debug_message_list.append(
            "\tQ(S_t+1, A_t+1) = " + str(q) + " + " + str(self.alpha_value) + " * (" + str(reward_value) + " + (" + str(self.gamma_value) + " * " + str(next_max_q) + ") - " + str(q) + ")"
        )
        self.debug_message_list.append(
            "\tQ(S_t+1, A_t+1) = " + str(new_q)
        )

    def __del__(self):
        '''
        デストラクタ
        デバッグモードの場合、デバッグメッセージをプリントする。
        '''
        if self.debug_mode is True:
            print("\n".join(self.debug_message_list))
