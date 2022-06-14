# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.policy_sampler import PolicySampler

import mxnet.ndarray as nd
import mxnet as mx
import numpy as np
import random

from accelbrainbase.observabledata._mxnet.lstm_networks import LSTMNetworks
from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm
from mxnet import gluon


class MarloPolicy(PolicySampler):
    '''
    Policy sampler for marlo.
    '''

    __action_dim = 5

    def get_action_dim(self):
        ''' getter '''
        return self.__action_dim
    
    def set_action_dim(self, value):
        ''' setter '''
        self.__action_dim = value

    action_dim = property(get_action_dim, set_action_dim)

    def __init__(
        self,
        env,
        norm_mode="z_score",
        scale=1.0,
        possible_n=10,
        ctx=mx.gpu(),
    ):
        '''
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - others : This class will not normalize the data.

        '''
        self.env = env
        self.__norm_mode = norm_mode
        self.__scale = scale
        self.__possible_n = possible_n
        self.__batch_size = 1
        self.__ctx = ctx

        self.reset()
        self.env.render()

        self.__reward_list = []

    def draw(self):
        '''
        Draw samples from distribtions.

        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        all_action_arr = None
        action_list = []
        state_feature_arr = self.pre_normalize(self.__state_arr)
        state_feature_arr = state_feature_arr.transpose((2, 0, 1))
        state_feature_arr = nd.expand_dims(state_feature_arr, axis=0)
        for n in range(self.__possible_n):
            if n < self.action_dim:
                action = n
            else:
                #action = self.env.action_space.sample()
                action = np.random.randint(low=0, high=self.action_dim)

            action_list.append(action)
            action_arr = nd.zeros(self.action_dim, ctx=self.__ctx)
            action_arr[action] = 1
            action_arr = nd.expand_dims(action_arr, axis=0)

            if all_action_arr is None:
                all_action_arr = nd.expand_dims(action_arr, axis=0)
            else:
                all_action_arr = nd.concat(
                    all_action_arr,
                    nd.expand_dims(action_arr, axis=0),
                    dim=0
                )

        return state_feature_arr, all_action_arr, np.array(action_list)

    def observe_state(self, state_arr, meta_data_arr):
        '''
        Observe states of agents in last epoch.

        Args:
            state_arr:      Tensor of state.
            meta_data_arr:  meta data of the state.
        '''
        self.__state_arr = nd.ndarray.array(meta_data_arr, ctx=state_arr.context)
        self.__state_meta_data_arr = meta_data_arr

    def observe_reward_value(
        self, 
        state_arr, 
        action_arr,
        meta_data_arr=None,
    ):
        '''
        Compute the reward value.
        
        Args:
            state_arr:              Tensor of state.
            action_arr:             Tensor of action.
            meta_data_arr:          Meta data of actions.

        Returns:
            Reward value.
        '''
        if state_arr is None:
            state_arr = self.__state_arr
        state_arr, reward_value_arr, done, info = self.env.step(meta_data_arr[0])
        self.__done = done
        self.__info = info
        self.__next_state_arr = state_arr

        self.__reward_list.append(reward_value_arr)

        reward_value_arr = nd.ndarray.array(np.array([reward_value_arr]), ctx=self.__ctx)
        return reward_value_arr

    def update_state(
        self, 
        action_arr, 
        meta_data_arr=None
    ):
        '''
        Update state.
        
        This method can be overrided for concreate usecases.

        Args:
            action_arr:     action in `self.t`.
            meta_data_arr:  meta data of the action.
        
        Returns:
            Tuple data.
            - state in `self.t+1`.
            - meta data of the state.
        '''
        state_arr = nd.ndarray.array(self.__next_state_arr, ctx=self.__ctx)
        return state_arr, self.__next_state_arr

    def check_the_end_flag(self, state_arr, meta_data_arr=None):
        '''
        Check the end flag.

        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_arr:      state in `self.t`.
            meta_data_arr:  meta data of the state.

        Returns:
            bool
        '''
        return self.__done

    def reset(self):
        '''
        Reset retro.
        '''
        meta_data_arr = self.env.reset()
        state_arr = nd.ndarray.array(meta_data_arr, ctx=self.__ctx)
        self.observe_state(state_arr=state_arr, meta_data_arr=meta_data_arr)

    def pre_normalize(self, arr):
        '''
        Normalize before observation.

        Args:
            arr:    Tensor.
        
        Returns:
            Tensor.
        '''
        if self.__norm_mode == "min_max":
            if arr.max() != arr.min():
                n = 0.0
            else:
                n = 1e-08
            arr = (arr - arr.min()) / (arr.max() - arr.min() + n)
        elif self.__norm_mode == "z_score":
            std = arr.asnumpy().std()
            if std == 0:
                std += 1e-08
            arr = (arr - arr.mean()) / std

        arr = arr * self.__scale
        return arr

    def get_reward_log_arr(self):
        ''' getter '''
        return np.array(self.__reward_list).reshape((len(self.__reward_list), ))

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    reward_log_arr = property(get_reward_log_arr, set_readonly)
