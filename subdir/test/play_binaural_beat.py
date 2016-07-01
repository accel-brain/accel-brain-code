#!/user/bin/env python
# -*- coding: utf-8 -*-
from PyBrainWave.brainbeat.binaural_beat import BinauralBeat
from PyBrainWave.waveform.sine_wave import SineWave


def main():
    wave_form = SineWave()
    brain_beat = BinauralBeat()
    brain_beat.wave_form = wave_form
    brain_beat.play_beat(frequencys=(300, 310), play_time=10, volume=0.5)

if __name__ == "__main__":
    main()
