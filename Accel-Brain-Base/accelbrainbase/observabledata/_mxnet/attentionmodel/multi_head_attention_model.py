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
from accelbrainbase.observabledata._mxnet.attention_model import AttentionModel


class MultiHeadAttentionModel(AttentionModel):
    '''
    Multi Head Attention Model.

    References:
        - Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
        - Floridi, L., & Chiriatti, M. (2020). GPT-3: Its nature, scope, limits, and consequences. Minds and Machines, 30(4), 681-694.
        - Miller, A., Fisch, A., Dodge, J., Karimi, A. H., Bordes, A., & Weston, J. (2016). Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126.
        - Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018) Improving Language Understanding by Generative Pre-Training. OpenAI (URL: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
        - Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.

    '''
    
    __head_n = 4

    def get_head_n(self):
        ''' getter '''
        return self.__head_n
    
    def set_head_n(self, value):
        ''' setter '''
        self.__head_n = value

    head_n = property(get_head_n, set_head_n)

    __seq_len = 1

    def get_seq_len(self):
        ''' getter '''
        return self.__seq_len
    
    def set_seq_len(self, value):
        ''' setter '''
        self.__seq_len = value

    seq_len = property(get_seq_len, set_seq_len)

    def inference(self, observed_arr, memory_arr, mask=None):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   Array like or sparse matrix as the observed data points.
            memory_arr:     Array like or sparse matrix as the observed data points.
            mask:   `mxnet.ndarray` of mask.

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        if mask is None:
            mask = nd.ones(
                shape=(
                    observed_arr.shape[0],
                    1,
                    1,
                ),
                ctx=observed_arr.context
            )
        return self(observed_arr, memory_arr, mask)

    def hybrid_forward(self, F, x, m, mask):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
            m:      `mxnet.ndarray` of observed data points. The shape is (batch_size, length of memory, depth).
            mask:   `mxnet.ndarray` of mask.

        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x, m, mask)

    def forward_propagation(self, F, x, m, mask):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points. The shape is (batch_size, length of query, depth).
            m:      `mxnet.ndarray` of observed data points. The shape is (batch_size, length of memory, depth).
            mask:   `mxnet.ndarray` of mask.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        query_arr = self.query_dense_layer(x)
        key_arr = self.key_dense_layer(m)
        value_arr = self.value_dense_layer(m)

        query_arr = F.reshape_like(
            query_arr,
            F.zeros(
                shape=(
                    self.batch_size, 
                    self.head_n,
                    self.seq_len, 
                    self.depth_dim // self.head_n
                ),
                ctx=self.ctx
            )
        )
        key_arr = F.reshape_like(
            key_arr,
            F.zeros(
                shape=(
                    self.batch_size, 
                    self.head_n,
                    self.seq_len, 
                    self.depth_dim // self.head_n
                ),
                ctx=self.ctx
            )
        )
        value_arr = F.reshape_like(
            value_arr,
            F.zeros(
                shape=(
                    self.batch_size, 
                    self.head_n,
                    self.seq_len, 
                    self.depth_dim // self.head_n
                ),
                ctx=self.ctx
            )
        )

        depth = self.depth_dim // self.head_n
        query_arr = query_arr * (depth ** -0.5)

        logit_arr = None
        for i in range(1, self.head_n+1):
            _query_arr = F.slice(query_arr, begin=(None, i-1), end=(None, i))
            _query_arr = F.reshape(_query_arr, (self.batch_size, self.seq_len, self.depth_dim//self.head_n))
            _key_arr = F.slice(key_arr, begin=(None, i-1), end=(None, i))
            _key_arr = F.reshape(_key_arr, (self.batch_size, self.seq_len, self.depth_dim//self.head_n))
            _logit_arr = F.batch_dot(
                _query_arr,
                _key_arr,
                transpose_b=True
            )
            if logit_arr is None:
                logit_arr = F.expand_dims(_logit_arr, axis=1)
            else:
                logit_arr = F.concat(
                    logit_arr,
                    F.expand_dims(_logit_arr, axis=1),
                    dim=1
                )

        if mask is not None and isinstance(mask, int) is False:
            logit_arr = F.broadcast_mul(logit_arr, mask)

        attention_weight_arr = F.softmax(logit_arr)
        attention_weight_arr = self.dropout(attention_weight_arr)

        attention_output_arr = None
        for i in range(1, self.head_n+1):
            _attention_weight_arr = F.slice(attention_weight_arr, begin=(None, i-1), end=(None, i))
            _attention_weight_arr = F.reshape(_attention_weight_arr, (self.batch_size, self.seq_len, self.seq_len))
            _value_arr = F.slice(value_arr, begin=(None, i-1), end=(None, i))
            _value_arr = F.reshape(_value_arr, (self.batch_size, self.seq_len, self.depth_dim//self.head_n))

            _attention_output_arr = F.batch_dot(_attention_weight_arr, _value_arr)

            if attention_output_arr is None:
                attention_output_arr = F.expand_dims(_attention_output_arr, axis=1)
            else:
                attention_output_arr = F.concat(
                    attention_output_arr,
                    F.expand_dims(_attention_output_arr, axis=1),
                    dim=1
                )

        attention_output_arr = F.reshape(
            attention_output_arr,
            (
                self.batch_size,
                self.seq_len,
                self.depth_dim
            )
        )
        output_arr = self.output_dense_layer(attention_output_arr)
        return output_arr
