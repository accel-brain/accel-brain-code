# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class ChordProgression(object):
    '''
    Chord Progression.
    '''
    
    __rule_df = pd.DataFrame(
        [
            ("I", "I"),
            ("I", "II"),
            ("I", "III"),
            ("I", "IV"),
            ("I", "V"),
            ("I", "VI"),
            ("II", "V"),
            ("III", "I"),
            ("III", "II"),
            ("III", "III"),
            ("III", "IV"),
            ("III", "V"),
            ("III", "VI"),
            ("IV", "I"),
            ("IV", "II"),
            ("IV", "III"),
            ("IV", "IV"),
            ("IV", "V"),
            ("V", "I"),
            ("V", "VI"),
            ("VI", "II"),
            ("VI", "III"),
            ("VI", "IV"),
            ("VI", "V"),
            ("VI", "VI")
        ],
        columns=["state", "action"]
    )

    def progression(self, octave, state):
        '''
        Progression.
        
        Args:
            octave:    Octave.
            state:     Now chord.
        '''
        df = self.__rule_df[self.__rule_df.state == state]
        return np.random.choice(df.action.values)
