# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.policy_sampler import PolicySampler
from accelbrainbase.iteratabledata._mxnet.unlabeled_t_hot_txt_iterator import UnlabeledTHotTXTIterator

import mxnet.ndarray as nd
import mxnet as mx
import numpy as np


class LabeledSummarizationPolicy(PolicySampler):
    '''
    Policy sampler for the Deep Q-learning to evaluate the value of the 
    "action" of selecting the image with the highest similarity based on 
    the "state" of observing an image.

    The state-action value is proportional to the similarity between the previously 
    observed image and the currently selected image.

    This class calculates the image similarity by mean squared error of images.
    '''

    # is-a `UnlabeledTHotTXTIterator`.
    __unlabeled_t_hot_txt_iterator = None

    def get_unlabeled_t_hot_txt_iterator(self):
        ''' getter for `UnlabeledTHotTXTIterator`. '''
        if isinstance(self.__unlabeled_t_hot_txt_iterator, UnlabeledTHotTXTIterator) is False:
            raise TypeError("The type of `__unlabeled_t_hot_txt_iterator` must be `UnlabeledTHotTXTIterator`.")
        return self.__unlabeled_t_hot_txt_iterator
    
    def set_unlabeled_t_hot_txt_iterator(self, value):
        ''' setter for `UnlabeledTHotTXTIterator`.'''
        if isinstance(value, UnlabeledTHotTXTIterator) is False:
            raise TypeError("The type of `__unlabeled_t_hot_txt_iterator` must be `UnlabeledTHotTXTIterator`.")
        self.__unlabeled_t_hot_txt_iterator = value

    unlabeled_t_hot_txt_iterator = property(get_unlabeled_t_hot_txt_iterator, set_unlabeled_t_hot_txt_iterator)

    def __init__(
        self,
        txt_path_list,
        abstract_pos="top",
        s_a_dist_weight=0.3
    ):
        txt_list = []
        for txt_path in txt_path_list:
            with open(txt_path) as f:
                txt_list.append(f.read())
        self.__txt_list = txt_list
        self.__abstract_pos = abstract_pos
        self.__s_a_dist_weight = s_a_dist_weight

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        observed_arr, label_arr = None, None
        for arr_tuple in self.unlabeled_t_hot_txt_iterator.generate_learned_samples():
            _meta_data_arr = self.unlabeled_t_hot_txt_iterator.token_arr[
                arr_tuple[0].asnumpy().argmax(axis=-1)
            ]

            if observed_arr is None:
                observed_arr = nd.expand_dims(arr_tuple[0], axis=1)
                meta_data_arr = np.expand_dims(_meta_data_arr, axis=1)
            else:
                observed_arr = nd.concat(
                    observed_arr, 
                    nd.expand_dims(arr_tuple[0], axis=1), 
                    dim=1
                )
                meta_data_arr = np.concatenate(
                    [
                        meta_data_arr,
                        np.expand_dims(_meta_data_arr, axis=1)
                    ],
                    axis=1
                )

            if observed_arr.shape[1] >= self.possible_n:
                break

        return observed_arr, meta_data_arr

    def observe_state(self, state_arr, meta_data_arr):
        '''
        Observe states of agents in last epoch.

        Args:
            state_arr:      Tensor of state.
            meta_data_arr:  meta data of the state.
        '''
        self.__state_arr = state_arr
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
        if state_arr is not None:
            t_hot_loss = -nd.mean(
                nd.flatten(state_arr) * nd.flatten(action_arr),
                axis=0, 
                exclude=True
            )
            reward_value_arr = t_hot_loss
            reward_value_arr = nd.expand_dims(reward_value_arr, axis=1)
        else:
            reward_value_arr = nd.zeros((
                action_arr.shape[0],
                1
            ), ctx=action_arr.context)

        if meta_data_arr is not None:
            add_reward_arr = nd.zeros((action_arr.shape[0], 1), ctx=action_arr.context)
            for batch in range(meta_data_arr.shape[0]):
                keyword = "".join(meta_data_arr[batch].reshape(1, -1).tolist()[0])
                reward = 0.0
                for i in range(len(self.__txt_list)):
                    key = self.__txt_list[i].index(keyword)
                    reward = reward + ((len(self.__txt_list[i]) - key) / len(self.__txt_list[i]))
                    reward = reward + (self.__txt_list[i].count(keyword) / len(self.__txt_list[i]))
                add_reward_arr[batch] = reward / len(self.__txt_list)
        else:
            add_reward_arr = nd.zeros((
                meta_data_arr.shape[0], 1), 
                ctx=meta_data_arr.context
            )

        reward_value_arr = (reward_value_arr * self.__s_a_dist_weight) + (add_reward_arr * (1 - self.__s_a_dist_weight))
        reward_value_arr = nd.tanh(reward_value_arr)
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
        return action_arr, meta_data_arr

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
        # As a rule, the learning can not be stopped.
        return False
