#!/user/bin/env python
# -*- coding: utf-8 -*-
from AccelBrainBeat.brainbeat.binaural_beat import BinauralBeat


def main():
    brain_beat = BinauralBeat()
    brain_beat.save_beat(
        output_file_name="save_binaural_beat.wav",
        frequencys=(300, 310),
        play_time=10,
        volume=0.01
    )

if __name__ == "__main__":
    main()
