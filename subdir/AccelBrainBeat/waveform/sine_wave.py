#!/user/bin/env python
# -*- coding: utf-8 -*-
import numpy
import math
from AccelBrainBeat.waveform.interface.wave_form_interface import WaveFormInterface


class SineWave(WaveFormInterface):
    '''
    インターフェイスの実現
    バイノーラルビートやモノラルビートで処理する対象となる
    正弦波の波形を計算するインターフェイス
    '''

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
        length = int(play_time * sample_rate)
        factor = float(frequency) * (math.pi * 2) / sample_rate
        return numpy.sin(numpy.arange(length) * factor)
