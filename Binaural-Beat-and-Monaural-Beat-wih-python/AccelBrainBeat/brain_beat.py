#!/user/bin/env python
# -*- coding: utf-8 -*-
import pyaudio
import numpy
import wave
from abc import ABCMeta, abstractmethod
from AccelBrainBeat.waveform.interface.wave_form_interface import WaveFormInterface
from AccelBrainBeat.waveform.sine_wave import SineWave


class BrainBeat(metaclass=ABCMeta):
    '''
    抽象クラス
    バイノーラルビートとモノラルビートの具象的差異を下位クラスで記述する
    Template Method Patternの構成
    波形部分のアルゴリズムが外部から委譲されることを前提とした
    Strategy Patternの構成
    '''

    # 波形部分のアルゴリズムの責任を担うインターフェイス
    __wave_form = None

    def get_wave_form(self):
        ''' getter '''
        if isinstance(self.__wave_form, WaveFormInterface) is False:
            raise TypeError()
        return self.__wave_form

    def set_wave_form(self, value):
        ''' setter '''
        if isinstance(value, WaveFormInterface) is False:
            raise TypeError()
        self.__wave_form = value

    # 波形部分のアルゴリズムの責任を担うインターフェイスのプロパティ
    wave_form = property(get_wave_form, set_wave_form)

    def __init__(self):
        '''
        初期化
        '''
        # デフォルトは正弦波
        self.wave_form = SineWave()

    def play_beat(
        self,
        frequencys,
        play_time,
        sample_rate=44100,
        volume=0.01
    ):
        '''
        引数で指定した条件でビートを鳴らす

        Args:
            frequencys:     (左の周波数(Hz), 右の周波数(Hz))のtuple
            play_time:      再生時間（秒）
            sample_rate:    サンプルレート
            volume:         音量

        Returns:
            void
        '''

        # 依存するライブラリの基底オブジェクト
        audio = pyaudio.PyAudio()
        # ストリーム
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=sample_rate,
            output=1
        )
        left_frequency, right_frequency = frequencys
        left_chunk = self.__create_chunk(left_frequency, play_time, sample_rate)
        right_chunk = self.__create_chunk(right_frequency, play_time, sample_rate)
        self.write_stream(stream, left_chunk, right_chunk, volume)
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def save_beat(
        self,
        output_file_name,
        frequencys,
        play_time,
        sample_rate=44100,
        volume=0.01
    ):
        '''
        引数で指定した条件でビートを鳴らす

        Args:
            frequencys:     (左の周波数(Hz), 右の周波数(Hz))のtuple
            play_time:      再生時間（秒）
            sample_rate:    サンプルレート
            volume:         音量

        Returns:
            void
        '''
        left_frequency, right_frequency = frequencys
        left_chunk = self.__create_chunk(left_frequency, play_time, sample_rate)
        right_chunk = self.__create_chunk(right_frequency, play_time, sample_rate)
        frame_list = self.read_stream(left_chunk, right_chunk, volume)

        wf = wave.open(output_file_name, 'wb')
        wf.setparams((2, 2, sample_rate, 0, 'NONE', 'not compressed'))
        wf.writeframes(b''.join(frame_list))
        wf.close()

    @abstractmethod
    def write_stream(self, stream, left_chunk, right_chunk, volume):
        '''
        抽象メソッド
        ビートを生成する

        Args:
            stream:         PyAudioのストリーム
            left_chunk:     左音源に対応するチャンク
            right_chunk:    右音源に対応するチャンク
            volume:         音量

        Returns:
            void
        '''
        raise NotImplementedError()

    @abstractmethod
    def read_stream(self, left_chunk, right_chunk, volume, bit16=32767.0):
        '''
        抽象メソッド
        wavファイルに保存するビートを読み込む

        Args:
            left_chunk:     左音源に対応するチャンク
            right_chunk:    右音源に対応するチャンク
            volume:         音量
            bit16:          整数化の条件

        Returns:
            フレームのlist
        '''
        raise NotImplementedError()

    def __create_chunk(self, frequency, play_time, sample_rate):
        '''
        チャンクを生成する

        Args:
            frequency:      周波数
            play_time:      再生時間（秒）
            sample_rate:    サンプルレート

        Returns:
            チャンクのnumpy配列
        '''
        chunks = []
        wave_form = self.wave_form.create(frequency, play_time, sample_rate)
        chunks.append(wave_form)
        chunk = numpy.concatenate(chunks)
        return chunk
