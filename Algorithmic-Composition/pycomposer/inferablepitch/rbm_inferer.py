# -*- coding: utf-8 -*-
import numpy as np
from pycomposer.inferable_pitch import InferablePitch
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine


class RBMInferer(InferablePitch):
    '''
    Inferacing next pitch by RBM.
    '''
    
    __pitch_rbm = None
    
    def get_pitch_rbm(self):
        ''' getter '''
        if isinstance(self.__pitch_rbm, RestrictedBoltzmannMachine) is False:
            raise TypeError()
        return self.__pitch_rbm

    def set_pitch_rbm(self, value):
        ''' setter '''
        if isinstance(value, RestrictedBoltzmannMachine) is False:
            raise TypeError()
        self.__pitch_rbm = value
    
    pitch_rbm = property(get_pitch_rbm, set_pitch_rbm)
    
    __inferancing_training_count = None
    
    def get_inferancing_training_count(self):
        ''' getter '''
        if isinstance(self.__inferancing_training_count, int) is False:
            raise TypeError()
        return self.__inferancing_training_count

    def set_inferancing_training_count(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError()
        self.__inferancing_training_count = value
    
    inferancing_training_count = property(get_inferancing_training_count, set_inferancing_training_count)
    
    __r_batch_size = 200
    
    def get_r_batch_size(self):
        ''' getter '''
        if isinstance(self.__r_batch_size, int) is False:
            raise TypeError()
        return self.__r_batch_size

    def set_r_batch_size(self, value):
        ''' setter '''
        if isinstance(value, int) is False:
            raise TypeError()
        self.__r_batch_size = value
    
    r_batch_size = property(get_r_batch_size, set_r_batch_size)
    
    def learn(self, tone_df, training_count=1, batch_size=200):
        '''
        Learning.
        
        Args:
            tone_df:            pd.DataFrame([], columns=["pitch", "and so on."])
            training_count:     Training count.
            batch_size:         The batch size of mini-batch training.
        '''
        for i in range(tone_df.shape[0]):
            pitch_arr = np.zeros(127)
            pitch_arr[tone_df.pitch.values[i]] = 1
            pitch_arr = pitch_arr.astype(np.float64)
            self.pitch_rbm.approximate_learning(
                pitch_arr,
                traning_count=training_count, 
                batch_size=batch_size
            )

    def inferance(self, pre_pitch, pitch_arr):
        '''
        Inferance and select next pitch of `pre_pitch` from the values of `pitch_arr`.
        
        Override.
        
        Args:
            pre_pitch:    The pitch in `t-1`.
            pitch_arr:    The list of selected pitchs.
        
        Returns:
            The pitch in `t`.
        '''
        test_arr = np.zeros(127)
        test_arr[pre_pitch] = 1
        test_arr = test_arr.astype(np.float64)
        self.pitch_rbm.approximate_inferencing(
            test_arr,
            traning_count=self.inferancing_training_count, 
            r_batch_size=self.r_batch_size
        )
        
        pitch = None
        for key in self.pitch_rbm.graph.visible_activity_arr.argsort()[::-1].tolist():
            if key in pitch_arr.tolist():
                pitch = key
                break

        if pitch is None:
            pitch = np.argmax(self.pitch_rbm.graph.visible_activity_arr)

        return int(pitch)
