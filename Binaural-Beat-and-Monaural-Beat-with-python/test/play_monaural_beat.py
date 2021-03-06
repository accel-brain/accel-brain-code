#!/user/bin/env python
# -*- coding: utf-8 -*-
from AccelBrainBeat.brainbeat.monaural_beat import MonauralBeat


def main():
    brain_beat = MonauralBeat()
    brain_beat.play_beat(
        frequencys=(300, 310),
        play_time=10,
        volume=0.01
    )

if __name__ == "__main__":
    main()
