# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pycomposer.inferable_duration import InferableDuration


class RBMTimeDiffInferer(InferableDuration):
    '''
    Inferacing duration by RBM.
    '''
    
    diff_df = None
    
    tone_df = None
    
    learning_rate = 0.00001
    
    beat_n = 4
    
    hidden_n = 100
    
    hidden_binary_flag = True
    
    inferancing_training_count = 1
    
    r_batch_size = 200
    
    rbm = None

    def learn(
        self, 
        training_count=1, 
        batch_size=200,
        metronome_time=60,
        measure_n=8,
        beat_n=8,
        total_measure_n=400
    ):
        '''
        Learning.
        
        Args:
            tone_df:            pd.DataFrame([], columns=["pitch", "and so on."])
            training_count:     Training count.
            batch_size:         The batch size of mini-batch training.
        '''
        tone_df = self.tone_df.sort_values(by=["start", "end"])

        chord_time = 0.0
        diff_tuple_list = []
        for measure in range(total_measure_n):
            measure += 1

            chord_start = chord_time
            chord_end = (((60/metronome_time) * measure_n * (measure))) + ((60/metronome_time) * (beat_n))

            chord_time = chord_end

            duration_arr = np.zeros(beat_n)
            for beat in range(beat_n):
                beat += 1
                if beat == 1 and measure != 1:
                    s_measure = measure - 1
                    s_beat = beat
                else:
                    s_measure = measure
                    s_beat = beat - 1

                v_start = (((60/metronome_time) * measure_n * (s_measure))) + ((60/metronome_time) * (s_beat))
                v_end = (((60/metronome_time) * measure_n * (measure))) + ((60/metronome_time) * (beat))
                
                df = tone_df[tone_df.start >= v_start]
                df = df[df.end <= v_end]
                duration_mean = df.duration.mean()
                duration_arr[beat-1] = duration_mean

            arr = duration_arr / (chord_end - chord_start)
            arr = arr.astype(np.float64)
            self.rbm.approximate_learning(
                arr,
                traning_count=training_count, 
                batch_size=batch_size
            )
        
        self.diff_df = pd.DataFrame(
            diff_tuple_list, 
            columns=[
                "measure",
                "beat",
                "v_start", 
                "v_end", 
                "duration_mean"
            ]
        )

    def inference(
        self,
        chord_time,
        measure,
        metronome_time,
        measure_n,
        beat_n,
        total_measure_n
    ):
        '''
        Inferance and select next pitch of `pre_pitch` from the values of `pitch_arr`.
        
        Override.
        
        Args:
            pre_pitch:    The pitch in `t-1`.
            pitch_arr:    The list of selected pitchs.
        
        Returns:
            The pitch in `t`.
        '''
        tone_df = self.tone_df.sort_values(by=["start", "end"])
        chord_start = chord_time
        chord_end = (((60/metronome_time) * measure_n * (measure))) + ((60/metronome_time) * (beat_n))

        chord_time = chord_end

        duration_rate = np.zeros(beat_n)
        for beat in range(beat_n):
            duration_arr = np.zeros(beat_n)

            if beat == 1 and measure != 1:
                s_measure = measure - 1
                s_beat = beat
            else:
                s_measure = measure
                s_beat = beat - 1

            v_start = (((60/metronome_time) * measure_n * (s_measure))) + ((60/metronome_time) * (s_beat))
            v_end = (((60/metronome_time) * measure_n * (measure))) + ((60/metronome_time) * (beat))

            df = tone_df[tone_df.start >= v_start]
            df = df[df.end <= v_end]
            duration_mean = df.duration.mean()
            duration_arr[beat-1] = duration_mean

        test_arr = duration_arr.astype(np.float64)
        self.rbm.approximate_inferencing(
            test_arr,
            traning_count=self.inferancing_training_count, 
            r_batch_size=self.r_batch_size
        )
        
        duration_arr = (chord_end - chord_start) * self.rbm.graph.visible_activity_arr
        return duration_arr
