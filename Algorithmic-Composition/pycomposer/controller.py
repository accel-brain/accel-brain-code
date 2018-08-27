# -*- coding: utf-8 -*-
import numpy as np
from pycomposer.rbmannealing.lstm_rtrbm_qmc import LSTMRTRBMQMC
from pycomposer.midi_vectorlizer import MidiVectorlizer
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
try:
    import matplotlib.pyplot as plt
except:
    pass

class Controller(object):
    '''
    Controller of composer.
    '''
    
    def __init__(self, verbose=False):
        '''
        Init.
        
        Args:
            verbose:    Verbose mode or not.
        '''
        logger = getLogger("pycomposer")
        handler = StreamHandler()
        if verbose is True:
            handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
        else:
            handler.setLevel(ERROR)
            logger.setLevel(ERROR)
            
        logger.addHandler(handler)

        logger = getLogger("pydbm")
        handler = StreamHandler()
        if verbose is True:
            handler.setLevel(DEBUG)
            logger.setLevel(DEBUG)
        else:
            handler.setLevel(ERROR)
            logger.setLevel(ERROR)
        logger.addHandler(handler)
    
    def compose(
        self,
        learned_midi_path,
        saved_midi_path,
        cycle_len=8,
        octave=7,
        batch_size=50,
        epoch=1,
        learning_rate=1e-05,
        hidden_num=100,
        annealing_cycles=100,
        program=0
    ):
        '''
        Execute composition.
        
        Args:
            learned_midi_path:  The file path which is extracted in learning.
            saved_midi_path:    Saved file path.
            cycle_len:          One cycle length observed by RBM as one sequencial data.
            octave:             The octave of music to be composed.
            epoch:              Epoch in RBM's mini-batch training.
            batch_size:         Batch size in RBM's mini-batch training.
            learning_rate:      Learning rate for RBM.
            hidden_num:         The number of units in hidden layer of RBM.
            annealing_cycles:   The number of cycles of annealing.
            program:            MIDI program number (instrument index), in [0, 127].

        '''
        midi_vectorlizer = MidiVectorlizer()
        midi_df = midi_vectorlizer.extract(learned_midi_path)
        midi_df = midi_df.sort_values(by=["start", "end"])
        midi_df = midi_df.dropna()

        rbm_annealing = LSTMRTRBMQMC()
        opt_df = rbm_annealing.compose(
            midi_df=midi_df,
            cycle_len=cycle_len,
            octave=octave,
            batch_size=batch_size,
            epoch=epoch,
            learning_rate=learning_rate,
            hidden_num=hidden_num,
            annealing_cycles=annealing_cycles
        )

        if opt_df[opt_df.start > 180].shape[0]:
            opt_df = opt_df[opt_df.start > 180]
            opt_df.start = opt_df.start - 180
            opt_df.end = opt_df.end - 180

        opt_df = opt_df[opt_df.duration < midi_df.duration.mean()]

        opt_df["program"] = program
        opt_df["pitch"] = opt_df["pitch"].astype(int)
        opt_df = opt_df[opt_df.pitch < 127]
        opt_df = opt_df[opt_df.pitch > 0]
        opt_df["velocity"] = opt_df["velocity"].astype(int)
        opt_df = opt_df[opt_df.velocity < 127]
        opt_df = opt_df[opt_df.velocity > 0]
        midi_vectorlizer.save(saved_midi_path, opt_df)
        
        self.__rbm_annealing = rbm_annealing
        
        print("MIDI file is created.")

    def visualize_cost(self):
        '''
        Visualize cost.
        '''
        fig = plt.figure(figsize=(20, 10))
        for i in range(10):
            plt.plot(
                self.__rbm_annealing.predicted_log_arr[self.__rbm_annealing.predicted_log_arr[:, 0] == i][:, 5],
                label="Trotter: " + str(i+1)
            )
        plt.title("Observed cost in annealing.")
        plt.legend()
        plt.show()
        plt.close()
