# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from logging import getLogger
import numpy as np
import pandas as pd
from pycomposer.rbm_cost_function import RBMCostFunction


class RBMAnnealing(metaclass=ABCMeta):
    '''
    Composer based on Restricted Boltzmann Machine and Annealing Model.
    '''
    
    def __init__(self, cycle_len=30):
        '''
        Init.

        '''
        logger = getLogger("pycomposer")
        self.__logger = logger

    @abstractmethod
    def create_rbm(self, visible_num, hidden_num, learning_rate):
        '''
        Build `RestrictedBoltzmmanMachine`.
        
        Args:
            visible_num:    The number of units in RBM's visible layer.
            hidden_num:     The number of units in RBM's hidden layer.
            learning_rate:  Learning rate.
        
        Returns:
            `RestrictedBoltzmmanMachine`.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def create_annealing(self, cost_functionable, params_arr, cycles_num=100):
        '''
        Build `AnnealingModel`.
        
        Args:
            cost_functionable:      is-a `CostFunctionable`.
            params_arr:             Random sampled parameters.
            cycles_num:             The number of annealing cycles.
        
        Returns:
            `AnnealingModel`.
        
        '''
        raise NotImplementedError("This method must be implemented.")

    def compose(
        self,
        midi_df,
        cycle_len=8,
        octave=5,
        epoch=100,
        batch_size=10,
        learning_rate=1e-05,
        hidden_num=100,
        annealing_cycles=100
    ):
        '''
        Init.
        
        Args:
            midi_df:            `pd.DataFrame` of MIDI file.
            cycle_len:          One cycle length observed by RBM as one sequencial data.
            octave:             The octave of music to be composed.
            epoch:              Epoch in RBM's mini-batch training.
            batch_size:         Batch size in RBM's mini-batch training.
            learning_rate:      Learning rate for RBM.
            hidden_num:         The number of units in hidden layer of RBM.
            annealing_cycles:   The number of cycles of annealing.

        Returns:
            `pd.DataFrame` of composed data.
        '''
        # Save only in learning.
        self.__max_pitch = midi_df.pitch.max()
        self.__min_pitch = midi_df.pitch.min()

        midi_df.pitch = midi_df.pitch % 12

        # Extract observed data points.
        observed_arr = self.__preprocess(midi_df, cycle_len)
        self.__logger.debug("The shape of observed data points: " + str(observed_arr.shape))

        rbm = self.create_rbm(
            visible_num=observed_arr.shape[-1], 
            hidden_num=hidden_num, 
            learning_rate=learning_rate
        )

        self.__logger.debug("Start RBM's learning.")
        # Learning.
        rbm.learn(
            # The `np.ndarray` of observed data points.
            observed_arr,
            # Training count.
            training_count=epoch, 
            # Batch size.
            batch_size=batch_size
        )

        self.__logger.debug("Start RBM's inferencing.")
        # Inference feature points.
        inferenced_arr = rbm.inference(
            observed_arr,
            training_count=1, 
            r_batch_size=-1
        )
        self.__logger.debug("The shape of inferenced data points: " + str(inferenced_arr.shape))

        cost_functionable = RBMCostFunction(inferenced_arr, rbm, batch_size)
        
        self.__logger.debug("Setup parameters.")
        params_arr = np.empty((annealing_cycles, observed_arr.shape[0], observed_arr.shape[1], observed_arr.shape[2]))
        generated_df_list = [None] * annealing_cycles
        for i in range(annealing_cycles):
            generated_df = self.__generate(midi_df)
            generated_df_list[i] = generated_df
            test_arr = self.__preprocess(generated_df, cycle_len)
            if test_arr.shape[0] > observed_arr.shape[0]:
                test_arr = test_arr[:observed_arr.shape[0]]
            elif test_arr.shape[0] < observed_arr.shape[0]:
                continue

            params_arr[i] = test_arr
        
        self.__logger.debug("The shape of parameters: " + str(params_arr.shape))

        annealing_model = self.create_annealing(cost_functionable, params_arr, cycles_num=annealing_cycles)

        self.__logger.debug("Start annealing.")
        # Execute annealing.
        annealing_model.annealing()

        opt_df = generated_df_list[
            int(annealing_model.predicted_log_arr[
                annealing_model.predicted_log_arr[:, 5] == annealing_model.predicted_log_arr[:, 5].min()
            ][0][4])
        ]
        
        opt_df["pitch"] = 12 + (opt_df["pitch"] * octave)
        
        self.__predicted_log_arr = annealing_model.predicted_log_arr
        self.__generated_df_list = generated_df_list
        return opt_df

    def __preprocess(self, midi_df_, cycle_len):
        observed_arr_list = []
        midi_df = midi_df_.copy()
        midi_df = midi_df.dropna()

        midi_df.pitch = (midi_df.pitch - midi_df.pitch.min()) / (midi_df.pitch.max() - midi_df.pitch.min())
        midi_df.start = (midi_df.start - midi_df.start.min()) / (midi_df.start.max() - midi_df.start.min())
        midi_df.end = (midi_df.end - midi_df.end.min()) / (midi_df.end.max() - midi_df.end.min())
        midi_df.duration = (midi_df.duration - midi_df.duration.min()) / (midi_df.duration.max() - midi_df.duration.min())
        midi_df.velocity = (midi_df.velocity - midi_df.velocity.min()) / (midi_df.velocity.max() - midi_df.velocity.min())

        observed_arr = np.empty((midi_df.shape[0] - cycle_len, cycle_len, 4))
        for i in range(cycle_len, midi_df.shape[0]):
            df = midi_df.iloc[i-cycle_len:i, :]
            observed_arr[i-cycle_len] = df[["pitch", "start", "duration", "velocity"]].values 
        observed_arr = observed_arr.astype(np.float64)
        
        return observed_arr

    def __generate(self, midi_df):
        start_arr = np.random.normal(
            loc=midi_df["start"].mean(),
            scale=midi_df["start"].std(),
            size=midi_df.shape[0]
        )
        duration_arr = np.random.normal(
            loc=midi_df["duration"].mean(),
            scale=midi_df["duration"].std(),
            size=midi_df.shape[0]
        )
        duration_arr[duration_arr < 0] = duration_arr[duration_arr > 0].min()
        velocity_arr = np.random.normal(
            loc=midi_df["velocity"].mean(),
            scale=midi_df["velocity"].std(),
            size=midi_df.shape[0]
        )
        pitch_arr = np.random.normal(
            loc=midi_df["pitch"].mean(),
            scale=midi_df["pitch"].std(),
            size=midi_df.shape[0]
        )
        velocity_arr = (velocity_arr).astype(int)
        pitch_arr = (pitch_arr).astype(int)
        
        generated_df = pd.DataFrame(
            np.c_[
                pitch_arr,
                start_arr,
                duration_arr,
                velocity_arr
            ],
            columns=[
                "pitch",
                "start",
                "duration",
                "velocity"
            ]
        )
        generated_df["start"] = generated_df["start"] + (generated_df["start"].min() * -1)
        generated_df["end"] = generated_df["start"] + generated_df["duration"]

        generated_df.pitch = generated_df.pitch % 12

        generated_df = generated_df.sort_values(by=["start", "end"])
        generated_df = generated_df.dropna()
        return generated_df

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_predicted_log_arr(self):
        ''' getter '''
        return self.__predicted_log_arr

    predicted_log_arr = property(get_predicted_log_arr, set_readonly)

    def get_generated_df_list(self):
        ''' getter '''
        return self.__generated_df_list

    generated_df_list = property(get_generated_df_list, set_readonly)
