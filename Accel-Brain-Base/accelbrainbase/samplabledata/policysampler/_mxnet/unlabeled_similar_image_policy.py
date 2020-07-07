# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.policy_sampler import PolicySampler
from accelbrainbase.iteratabledata.unlabeled_image_iterator import UnlabeledImageIterator

from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
import mxnet.ndarray as nd
import mxnet as mx


class UnlabeledSimilarImagePolicy(PolicySampler):
    '''
    Policy sampler for the Deep Q-learning to evaluate the value of the 
    "action" of selecting the image with the highest similarity based on 
    the "state" of observing an image.

    The state-action value is proportional to the similarity between the previously 
    observed image and the currently selected image.

    This class calculates the image similarity by mean squared error of images.
    '''

    # is-a `UnlabeledImageIterator`.
    __unlabeled_image_iterator = None

    def get_unlabeled_image_iterator(self):
        ''' getter for `UnlabeledImageIterator`. '''
        if isinstance(self.__unlabeled_image_iterator, UnlabeledImageIterator) is False:
            raise TypeError("The type of `__unlabeled_image_iterator` must be `UnlabeledImageIterator`.")
        return self.__unlabeled_image_iterator
    
    def set_unlabeled_image_iterator(self, value):
        ''' setter for `UnlabeledImageIterator`.'''
        if isinstance(value, UnlabeledImageIterator) is False:
            raise TypeError("The type of `__unlabeled_image_iterator` must be `UnlabeledImageIterator`.")
        self.__unlabeled_image_iterator = value

    unlabeled_image_iterator = property(get_unlabeled_image_iterator, set_unlabeled_image_iterator)

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        observed_arr, label_arr = None, None
        for arr_tuple in self.unlabeled_image_iterator.generate_learned_samples():
            if observed_arr is None:
                observed_arr = nd.expand_dims(arr_tuple[0], axis=1)
            else:
                observed_arr = nd.concat(
                    observed_arr, 
                    nd.expand_dims(arr_tuple[0], axis=1), 
                    dim=1
                )

            if observed_arr.shape[1] >= self.possible_n:
                break

        return observed_arr, None

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
            mse_arr = nd.mean(
                nd.square(
                    nd.flatten(state_arr),
                    nd.flatten(action_arr)
                ),
                axis=0, 
                exclude=True
            )
            reward_value_arr = 1 / mse_arr
            reward_value_arr = nd.expand_dims(reward_value_arr, axis=1)
        else:
            reward_value_arr = nd.zeros((
                action_arr.shape[0],
                1
            ), ctx=action_arr.context)

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
