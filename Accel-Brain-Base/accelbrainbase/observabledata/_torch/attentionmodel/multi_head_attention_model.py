# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
import numpy as np
from logging import getLogger
import torch
from accelbrainbase.observabledata._torch.attention_model import AttentionModel


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

    def __init__(
        self,
        depth_dim,
        computable_loss,
        initializer_f=None,
        optimizer_f=None,
        dropout_rate=0.5,
        learning_rate=1e-05,
        ctx="cpu",
        regularizatable_data_list=[],
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            depth_dim:                      `int` of dimension of dense layer.
            computable_loss:                is-a `ComputableLoss` or `gluon.loss.Loss`.
            dropout_rate:                   `float` of dropout rate.
            initializer:                    is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.

            optimizer_name:                 `str` of name of optimizer.
            ctx:                            `mx.cpu()` or `mx.gpu()`.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            regularizatable_data_list:           `list` of `RegularizatableData`.
            not_init_flag:                  `bool` of whether initialize parameters or not.
        '''
        super().__init__(
            depth_dim=depth_dim,
            computable_loss=computable_loss,
            initializer_f=initializer_f,
            optimizer_f=optimizer_f,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            ctx=ctx,
            regularizatable_data_list=regularizatable_data_list,
            not_init_flag=not_init_flag,
        )

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
        return self(observed_arr, memory_arr, mask)

    def forward(self, x, m, mask=None):
        '''
        Forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
            m:      `mxnet.ndarray` of observed data points. The shape is (batch_size, length of memory, depth).
            mask:   `mxnet.ndarray` of mask.

        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        if mask is None:
            mask = torch.ones(
                (
                    x.shape[0],
                    1,
                    1,
                )
            )
            mask = mask.to(x.device)

        self.batch_size = x.shape[0]
        self.__seq_len = x.shape[1]
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, axis=1)
        elif len(x.shape) > 3:
            x = x.reshape((
                self.batch_size,
                self.__seq_len,
                -1
            ))

        m = m.reshape_as(x)

        self.initialize_params(
            input_dim=x.shape[2]
        )

        query_arr = self.query_dense_layer(x)
        key_arr = self.key_dense_layer(m)
        value_arr = self.value_dense_layer(m)

        query_arr = query_arr.reshape(
            (
                self.batch_size, 
                self.head_n,
                self.seq_len, 
                self.depth_dim // self.head_n
            )
        )
        key_arr = key_arr.reshape(
            (
                self.batch_size, 
                self.head_n,
                self.seq_len, 
                self.depth_dim // self.head_n
            )
        )
        value_arr = value_arr.reshape(
            (
                self.batch_size, 
                self.head_n,
                self.seq_len, 
                self.depth_dim // self.head_n
            )
        )

        depth = self.depth_dim // self.head_n
        query_arr = query_arr * (depth ** -0.5)

        logit_arr = None
        for i in range(1, self.head_n+1):
            _query_arr = query_arr[:, i-1:i]
            _query_arr = _query_arr.reshape((self.batch_size, self.seq_len, self.depth_dim//self.head_n))
            _key_arr = key_arr[:, i-1:i]
            _key_arr = _key_arr.reshape((self.batch_size, self.seq_len, self.depth_dim//self.head_n))

            _logit_arr = torch.bmm(
                _query_arr,
                _key_arr.reshape((
                    _key_arr.shape[0],
                    _key_arr.shape[2],
                    _key_arr.shape[1]
                ))
            )
            if logit_arr is None:
                logit_arr = torch.unsqueeze(_logit_arr, axis=1)
                #logit_arr = _logit_arr
            else:
                logit_arr = torch.cat(
                    (
                        logit_arr,
                        torch.unsqueeze(_logit_arr, axis=1)
                    ),
                    dim=1
                )
                #logit_arr = torch.cat((logit_arr, _logit_arr), axis=1)

        if mask is not None and isinstance(mask, int) is False:
            logit_arr = torch.mul(logit_arr, mask)

        attention_weight_arr = self.softmax(logit_arr)
        attention_weight_arr = self.dropout(attention_weight_arr)

        attention_output_arr = None
        for i in range(1, self.head_n+1):
            _attention_weight_arr = attention_weight_arr[:, i-1:i]
            _attention_weight_arr = _attention_weight_arr.reshape((
                self.batch_size, self.seq_len, self.seq_len
            ))
            _value_arr = value_arr[:, i-1:i]
            _value_arr = _value_arr.reshape((
                self.batch_size, self.seq_len, self.depth_dim//self.head_n
            ))
            _attention_output_arr = torch.bmm(
                _attention_weight_arr, 
                _value_arr
            )

            if attention_output_arr is None:
                attention_output_arr = torch.unsqueeze(_attention_output_arr, axis=1)
            else:
                attention_output_arr = torch.cat(
                    (
                        attention_output_arr,
                        torch.unsqueeze(_attention_output_arr, axis=1)
                    ),
                    dim=1
                )

        attention_output_arr = attention_output_arr.reshape(
            (
                self.batch_size,
                self.seq_len,
                self.depth_dim
            )
        )
        output_arr = self.output_dense_layer(attention_output_arr)
        return output_arr
