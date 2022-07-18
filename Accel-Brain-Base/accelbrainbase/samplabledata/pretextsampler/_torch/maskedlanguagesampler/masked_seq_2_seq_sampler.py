# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.pretextsampler._torch.masked_language_sampler import MaskedLanguageSampler
import numpy as np
import torch


class MaskedSeq2SeqSampler(MaskedLanguageSampler):
    '''
    Sampler for pretext task based on the 
    Masked Sequence to Sequence Pre-training for Language Generation.

    References:
        - Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2019). Mass: Masked sequence to sequence pre-training for language generation. arXiv preprint arXiv:1905.02450.
    '''

    def __init__(
        self,
        masked_symbol=-1000,
        masked_seq_len=5,
    ):
        '''
        Init.

        Args:
            masked_symbol:      Symbol of mask.
            masked_seq_len:     The length of mask.
        '''
        self.masked_symbol = masked_symbol
        self.__masked_seq_len = masked_seq_len

    def preprocess(self, target_domain_arr):
        '''
        Preprocess observed data points in target domain.

        Args:
            target_domain_arr:      Tensor of observed data points in target domain.
                                    The rank of tensor is more than 2.
                                    The shape is (batch, sequence, token vector, ...).
        '''
        if self.__masked_seq_len == 1:
            super().preprocess(target_domain_arr)
            return

        pretext_label_arr = None
        target_domain_arr = target_domain_arr.float()
        pretext_encoded_observed_arr = target_domain_arr.detach()
        pretext_decoded_observed_arr = target_domain_arr.detach()
        for batch in range(target_domain_arr.shape[0]):
            target_seq = np.random.randint(
                low=0,
                high=target_domain_arr[batch].shape[0] - self.__masked_seq_len
            )
            if pretext_label_arr is None:
                pretext_label_arr = torch.unsqueeze(
                    target_domain_arr[batch, target_seq:target_seq+self.__masked_seq_len],
                    axis=0
                )
            else:
                pretext_label_arr = torch.cat(
                    (
                        pretext_label_arr,
                        torch.unsqueeze(
                            target_domain_arr[batch, target_seq:target_seq+self.__masked_seq_len],
                            axis=0
                        ),
                    ),
                    dim=0
                )
            target_domain_arr[batch, target_seq:target_seq+self.__masked_seq_len] = self.masked_symbol

        pretext_decoded_observed_arr = pretext_decoded_observed_arr * (target_domain_arr != self.masked_symbol).to(torch.int32) * self.masked_symbol

        self.pretext_encoded_observed_arr = target_domain_arr.float()
        self.pretext_decoded_observed_arr = pretext_decoded_observed_arr.float()
        self.pretext_encoded_mask_arr = None
        self.pretext_decoded_mask_arr = None
        self.pretext_label_arr = pretext_label_arr
