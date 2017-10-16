#!/user/bin/env python
# -*- coding: utf-8 -*-
import argparse
from AccelBrainBeat.brainbeat.binaural_beat import BinauralBeat


def main(params):
    brain_beat = BinauralBeat()
    brain_beat.play_beat(
        frequencys=(params["left"], params["right"]),
        play_time=params["time"],
        volume=params["volume"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create and play the Binaural Beat.'
    )
    parser.add_argument(
        '-l',
        '--left',
        type=int,
        default=400,
        help='Left frequencys (Hz).'
    )
    parser.add_argument(
        '-r',
        '--right',
        type=int,
        default=430,
        help='Right frequencys (Hz).'
    )

    parser.add_argument(
        '-t',
        '--time',
        type=int,
        default=10,
        help='Play time. This is per seconds.'
    )

    parser.add_argument(
        '-v',
        '--volume',
        type=float,
        default=0.1,
        help='Sound volume.'
    )
    args = parser.parse_args()
    params = vars(args)
    main(params)
