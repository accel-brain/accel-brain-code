#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class MazeQLearningInterface(metaclass=ABCMeta):
    '''
    Q学習で迷路探索を実装する場合の
    下位クラスのメソッドを規定する

    抽象メソッドのみを定義した抽象クラスとして配備することで、
    本クラスを事実上のインターフェイスとして利用する
    '''

    @abstractmethod
    def initialize(self, square_map_data):
        '''
        迷路を初期化する
        文字列の迷路マップデータ：square_map_dataを2次元のリスト：map_data_listに格納する。

        Args:
            square_map_data:      n×nの二次元リストに格納されることを前提としたCSV形式の文字列

        '''
        raise NotImplementedError()

    @abstractmethod
    def decide_start_point(self, map_data_list):
        '''
        マップデータ内のStart地点:Sの座標を返す。

        Args:
            map_data_list:      迷路のマップデータ。n×nの二次元リスト。

        Returns:
            マップデータ内のStart地点:Sの座標：(x, y)のtuple

        Exceptions:
            ValueError:         迷路のマップデータにスタート地点：Sが無い場合に発生する例外。

        '''
        raise NotImplementedError()

    @abstractmethod
    def extract_reward_value(self, point):
        '''
        指定した座標のfieldの値を返す. エピソード終了判定もする。

        Args:
            point:      現在の位置座標

        Returns:
            (報酬, 終了判定)のtuple

        Exceptions:
            ValueError:   エージェントが壁にぶつかった場合に発生する例外。

        '''
        raise NotImplementedError()

    @abstractmethod
    def debug_map_data_list(self, state_key=None, action_key=None, i=0, end_flag=False):
        '''
        マップ情報や現在位置の情報をデバッグメッセージに追加する。

        Args:
            state_key:      現在位置(x, y)のtuple
            action_key:     次回行動時の位置(x, y)のtuple
            end_flag:       学習完了フラグ

        '''
        raise NotImplemented()
