# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.pretext_sampler import PretextSampler
import numpy as np
import torch


class SentenceOrderSampler(PretextSampler):
    '''
    Sampler for pretext task based on the sentence order prediction problem.

    References:
        - Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.
    '''

    def __init__(
        self,
        sentence_flip_prob=0.5,
    ):
        '''
        Init.

        Args:
            sentence_flip_prob:      The probability of sentence filp.
        '''
        self.__sentence_flip_prob = sentence_flip_prob

    def preprocess(self, target_domain_arr):
        '''
        Preprocess observed data points in target domain.

        Args:
            target_domain_arr:      Tensor of observed data points in target domain.
                                    The rank of tensor is more than 2.
                                    The shape is (batch, sequence, token vector, ...).
        '''
        pretext_label_list = []
        pretext_encoded_observed_arr = target_domain_arr.detach().copy()
        pretext_decoded_observed_arr = target_domain_arr.detach().copy()
        for batch in range(pretext_encoded_observed_arr.shape[0]):
            pretext_label = np.random.binomial(1, self.__sentence_flip_prob, 1)[0]
            pretext_label_list.append(pretext_label)
            if pretext_label == 1:
                target_seq = np.random.randint(
                    low=1,
                    high=pretext_encoded_observed_arr[batch].shape[0] - 1
                )
                before_seq_arr = pretext_encoded_observed_arr[batch][:target_seq]
                after_seq_arr = pretext_encoded_observed_arr[batch][target_seq:]

                pretext_encoded_observed_arr[batch] = torch.cat(
                    (
                        after_seq_arr,
                        before_seq_arr
                    ),
                    dim=0
                )

        pretext_label_arr = np.zeros((len(pretext_label_list), 2))
        for i in range(len(pretext_label_list)):
            pretext_label_arr[i, pretext_label_list[i]] = 1

        pretext_label_arr = torch.from_numpy(
            pretext_label_arr
        )
        pretext_label_arr = pretext_label_arr.to(target_domain_arr.device)

        self.pretext_encoded_observed_arr = pretext_encoded_observed_arr
        self.pretext_decoded_observed_arr = pretext_decoded_observed_arr
        self.pretext_encoded_mask_arr = None
        self.pretext_decoded_mask_arr = None
        self.pretext_label_arr = pretext_label_arr.float()
