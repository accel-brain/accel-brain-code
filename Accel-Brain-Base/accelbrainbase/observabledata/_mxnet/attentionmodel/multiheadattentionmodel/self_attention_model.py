# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import MXNetError
from logging import getLogger
from accelbrainbase.observabledata._mxnet.attentionmodel.multi_head_attention_model import MultiHeadAttentionModel


class SelfAttentionModel(MultiHeadAttentionModel):
    '''
    Self-attention Model.

    References:
        - Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
        - Floridi, L., & Chiriatti, M. (2020). GPT-3: Its nature, scope, limits, and consequences. Minds and Machines, 30(4), 681-694.
        - Miller, A., Fisch, A., Dodge, J., Karimi, A. H., Bordes, A., & Weston, J. (2016). Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126.
        - Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018) Improving Language Understanding by Generative Pre-Training. OpenAI (URL: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
        - Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.

    '''
    
    def inference(self, observed_arr, mask=None):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   Array like or sparse matrix as the observed data points.
            mask:   `mxnet.ndarray` of mask.

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return super().inference(observed_arr, observed_arr, mask)
