#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class WaveFormInterface(metaclass=ABCMeta):
    '''
    抽象クラスを事実上のインターフェイスとする

    バイノーラルビートやモノラルビートで処理する対象となる
    音の波形を計算するインターフェイス

    通常ならば正弦波となる
    '''

    @abstractmethod
    def create(self, frequency, play_time, sample_rate):
        '''
        音の波形を生成する

        Args:
            frequency:      周波数
            play_time:      再生時間
            sample_rate:    サンプルレート

        Returns:
            波形要素を格納した配列
        '''
        raise NotImplementedError()
