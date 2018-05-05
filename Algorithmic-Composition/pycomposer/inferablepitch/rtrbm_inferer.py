# -*- coding: utf-8 -*-
import numpy as np
from pycomposer.inferable_pitch import InferablePitch
from pydbm.dbm.builders.rt_rbm_simple_builder import RTRBMSimpleBuilder
from pydbm.approximation.rt_rbm_cd import RTRBMCD
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.softmax_function import SoftmaxFunction


class RTRBMInferer(InferablePitch):
    '''
    Inferacing next pitch by RTRBM.
    '''
    
    def __init__(
        self,
        learning_rate=0.00001,
        hidden_n=100,
        hidden_binary_flag=True,
        inferancing_training_count=1,
        r_batch_size=200
    ):
        self.__inferancing_training_count = inferancing_training_count
        self.__r_batch_size = r_batch_size

        # `Builder` in `Builder Pattern` for RTRBM.
        rtrbm_builder = RTRBMSimpleBuilder()
        # Learning rate.
        rtrbm_builder.learning_rate = learning_rate
        # Set units in visible layer.
        rtrbm_builder.visible_neuron_part(
            SoftmaxFunction(), 
            127
        )
        # Set units in hidden layer.
        rtrbm_builder.hidden_neuron_part(
            LogisticFunction(normalize_flag=False, binary_flag=hidden_binary_flag), 
            hidden_n
        )
        # Set units in RNN layer.
        rtrbm_builder.rnn_neuron_part(
            LogisticFunction(normalize_flag=False, binary_flag=False)
        )
        # Set graph and approximation function.
        rtrbm_builder.graph_part(RTRBMCD())
        # Building.
        self.__pitch_rbm = rtrbm_builder.get_result()

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
            self.__pitch_rbm.approximate_learning(
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
        self.__pitch_rbm.approximate_inferencing(
            test_arr,
            traning_count=self.__inferancing_training_count, 
            r_batch_size=self.__r_batch_size
        )
        
        pitch = None
        for key in self.__pitch_rbm.graph.visible_activity_arr.argsort()[::-1].tolist():
            if key in pitch_arr.tolist():
                pitch = key
                break

        if pitch is None:
            pitch = np.argmax(self.__pitch_rbm.graph.visible_activity_arr)

        return int(pitch)
