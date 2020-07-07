# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.policy_sampler import PolicySampler
from accelbrainbase.iteratabledata.labeled_image_iterator import LabeledImageIterator
import mxnet.ndarray as nd
import mxnet as mx


class LabeledSimilarImagePolicy(PolicySampler):
    '''
    Policy sampler for the Deep Q-learning to evaluate the value of the 
    "action" of selecting the image with the highest similarity based on 
    the "state" of observing an image.

    The state-action value is proportional to the similarity between the previously 
    observed image and the currently selected image.

    This class calculates the image similarity by cross entorpy of labels (metadata).
    '''

    # Meta data of states.
    __state_meta_data_arr = None

    # is-a `LabeledImageIterator`.
    __labeled_image_iterator = None

    def get_labeled_image_iterator(self):
        ''' getter for `LabeledImageIterator`.'''
        if isinstance(self.__labeled_image_iterator, LabeledImageIterator) is False:
            raise TypeError("The type of `labeled_image_iterator` must be `LabeledImageIterator`.")
        return self.__labeled_image_iterator
    
    def set_labeled_image_iterator(self, value):
        ''' setter for `LabeledImageIterator`.'''
        if isinstance(value, LabeledImageIterator) is False:
            raise TypeError("The type of `labeled_image_iterator` must be `LabeledImageIterator`.")
        self.__labeled_image_iterator = value

    labeled_image_iterator = property(get_labeled_image_iterator, set_labeled_image_iterator)

    __computable_loss = None

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        if self.__computable_loss is None:
            self.__computable_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(
                sparse_label=False
            )

        observed_arr, label_arr = None, None
        for arr_tuple in self.labeled_image_iterator.generate_learned_samples():
            if observed_arr is None:
                observed_arr = nd.expand_dims(arr_tuple[0], axis=1)
            else:
                observed_arr = nd.concat(
                    observed_arr, 
                    nd.expand_dims(arr_tuple[0], axis=1), 
                    dim=1
                )
            if label_arr is None:
                label_arr = nd.expand_dims(arr_tuple[1], axis=1)
            else:
                label_arr = nd.concat(
                    label_arr, 
                    nd.expand_dims(arr_tuple[1], axis=1),
                    dim=1
                )

            if observed_arr.shape[1] >= self.possible_n:
                break

        return observed_arr, label_arr

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
        if self.__state_meta_data_arr is not None:
            cross_entropy_arr = self.__computable_loss(
                meta_data_arr,
                self.__state_meta_data_arr
            )
            cross_entropy_arr = nd.expand_dims(cross_entropy_arr, axis=1)
            reward_value_arr = 1 / cross_entropy_arr
        else:
            reward_value_arr = nd.zeros((
                meta_data_arr.shape[0], 1), 
                ctx=meta_data_arr.context
            )

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
