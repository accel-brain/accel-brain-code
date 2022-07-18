# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.pretext_sampler import PretextSampler
import numpy as np
import torch


class MaskedLanguageSampler(PretextSampler):
    '''
    Sampler for pretext task based on the Masked Language Modeling.

    References:
        - Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
        - Taylor, W. L. (1953). “Cloze procedure”: A new tool for measuring readability. Journalism quarterly, 30(4), 415-433.
    '''

    def __init__(
        self,
        masked_symbol=-1000,
    ):
        '''
        Init.

        Args:
            masked_symbol:      Symbol of mask.
        '''
        self.masked_symbol = masked_symbol

    def preprocess(self, target_domain_arr):
        '''
        Preprocess observed data points in target domain.

        Args:
            target_domain_arr:      Tensor of observed data points in target domain.
                                    The rank of tensor is more than 2.
                                    The shape is (batch, sequence, token vector, ...).
        '''
        pretext_label_arr = None
        target_domain_arr = target_domain_arr.float()
        pretext_encoded_observed_arr = target_domain_arr.detach()
        pretext_decoded_observed_arr = torch.ones_like(target_domain_arr) * self.masked_symbol
        for batch in range(target_domain_arr.shape[0]):
            target_seq = np.random.randint(
                low=0,
                high=target_domain_arr[batch].shape[0]
            )
            if pretext_label_arr is None:
                pretext_label_arr = torch.unsqueeze(
                    target_domain_arr[batch, target_seq],
                    axis=0
                )
            else:
                pretext_label_arr = torch.cat(
                    (
                        pretext_label_arr,
                        torch.unsqueeze(
                            target_domain_arr[batch, target_seq],
                            axis=0
                        )
                    ),
                    dim=0
                )
            target_domain_arr[batch, target_seq] = self.masked_symbol

        self.pretext_encoded_observed_arr = target_domain_arr.float()
        self.pretext_decoded_observed_arr = pretext_decoded_observed_arr.float()
        self.pretext_encoded_mask_arr = None
        self.pretext_decoded_mask_arr = None
        self.pretext_label_arr = pretext_label_arr.float()
