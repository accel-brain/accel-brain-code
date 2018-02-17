#!/user/bin/env python
# -*- coding: utf-8 -*-
import argparse
from AccelBrainBeat.brainbeat.binaural_beat import BinauralBeat


def main(params, default_file_name):
    if params["output_file_name"] == default_file_name:
        params["output_file_name"] = params["output_file_name"].replace("{-l}", str(params["left"]))
        params["output_file_name"] = params["output_file_name"].replace("{-r}", str(params["right"]))
        params["output_file_name"] = params["output_file_name"].replace("{-t}", str(params["time"]))
        params["output_file_name"] = params["output_file_name"].replace("{-v}", str(params["volume"]))

    print("Created file: " + params["output_file_name"])

    brain_beat = BinauralBeat()
    brain_beat.save_beat(
        output_file_name=params["output_file_name"],
        frequencys=(params["left"], params["right"]),
        play_time=params["time"],
        volume=params["volume"]
    )

if __name__ == "__main__":
    default_file_name = "binaural_beat_{-l}_{-r}_{-t}_{-v}.wav"

    parser = argparse.ArgumentParser(
        description='Create the Binaural Beat and save wav file.'
    )
    parser.add_argument(
        '-o',
        '--output_file_name',
        type=str,
        default=default_file_name,
        help='Output file name.'
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
    main(params, default_file_name)
