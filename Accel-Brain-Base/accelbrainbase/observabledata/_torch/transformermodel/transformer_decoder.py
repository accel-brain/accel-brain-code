# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.transformer_model import TransformerModel
from accelbrainbase.observabledata._torch.attention_model import AttentionModel
from accelbrainbase.observabledata._torch.attentionmodel.multi_head_attention_model import MultiHeadAttentionModel
from accelbrainbase.observabledata._torch.attentionmodel.multiheadattentionmodel.self_attention_model import SelfAttentionModel
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase.regularizatable_data import RegularizatableData
import numpy as np
from logging import getLogger
import torch
from torch import nn
from torch.optim.adamw import AdamW


class TransformerDecoder(nn.Module, TransformerModel):
    '''
    Decoder of Transformer.

    References:
        - Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
        - Floridi, L., & Chiriatti, M. (2020). GPT-3: Its nature, scope, limits, and consequences. Minds and Machines, 30(4), 681-694.
        - Miller, A., Fisch, A., Dodge, J., Karimi, A. H., Bordes, A., & Weston, J. (2016). Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126.
        - Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018) Improving Language Understanding by Generative Pre-Training. OpenAI (URL: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
        - Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.

    '''

    __batch_size = None

    def get_batch_size(self):
        ''' getter '''
        return self.__batch_size
    
    def set_batch_size(self, value):
        ''' setter '''
        self.__batch_size = value
        for i in range(len(self.self_attention_list)):
            if self.self_attention_list[i] is not None:
                self.self_attention_list[i].batch_size = value
        for i in range(len(self.multi_head_attention_list)):
            if self.multi_head_attention_list[i] is not None:
                self.multi_head_attention_list[i].batch_size = value

    batch_size = property(get_batch_size, set_batch_size)

    __head_n = None

    def get_head_n(self):
        ''' getter '''
        return self.__head_n
    
    def set_head_n(self, value):
        ''' setter '''
        self.__head_n = value
        for i in range(len(self.self_attention_list)):
            if self.self_attention_list[i] is not None:
                self.self_attention_list[i].head_n = value
        for i in range(len(self.multi_head_attention_list)):
            if self.multi_head_attention_list[i] is not None:
                self.multi_head_attention_list[i].head_n = value

    head_n = property(get_head_n, set_head_n)

    __seq_len = None

    def get_seq_len(self):
        ''' getter '''
        return self.__seq_len
    
    def set_seq_len(self, value):
        ''' setter '''
        self.__seq_len = value
        for i in range(len(self.self_attention_list)):
            if self.self_attention_list[i] is not None:
                self.self_attention_list[i].seq_len = value
        for i in range(len(self.multi_head_attention_list)):
            if self.multi_head_attention_list[i] is not None:
                self.multi_head_attention_list[i].seq_len = value

    seq_len = property(get_seq_len, set_seq_len)

    __depth_dim = None

    def get_depth_dim(self):
        ''' getter '''
        return self.__depth_dim
    
    def set_depth_dim(self, value):
        ''' setter '''
        self.__depth_dim = value
        for i in range(len(self.self_attention_list)):
            if self.self_attention_list[i] is not None:
                self.self_attention_list[i].depth_dim = value
        for i in range(len(self.multi_head_attention_list)):
            if self.multi_head_attention_list is not None:
                self.multi_head_attention_list[i].depth_dim = value

    depth_dim = property(get_depth_dim, set_depth_dim)

    def __init__(
        self,
        depth_dim,
        layer_n,
        computable_loss,
        optimizer_f=None,
        head_n=3,
        self_attention_layer_norm_list=[],
        multi_head_attention_layer_norm_list=[],
        fc_layer_norm_list=[],
        filter_fc_list=[],
        dropout_rate=0.1,
        fc_list=[],
        self_attention_activation_list=[],
        multi_head_attention_activation_list=[],
        fc_activation_list=[],
        output_layer_norm=None,
        output_fc=None,
        output_dim=None,
        learning_rate=6e-06,
        weight_decay=0.01,
        regularizatable_data_list=[],
        ctx="cpu",
        not_init_flag=False,
    ):
        """
        Init.

        Args:
            computable_loss:                is-a `ComputableLoss` or `gluon.loss`.
            encoder:                        is-a `TransformerModel`.
            decoder:                        is-a `TransformerModel`.
            layer_n:                        `int` of the number of layers.
            head_n:                         `int` of the number of heads for multi-head attention model.
            seq_len:                        `int` of the length of sequences.
            depth_dim:                      `int` of dimension of dense layer.
            hidden_dim:                     `int` of dimension of hidden(encoder) layer.
            self_attention_activation_list: `list` of `str` of activation function for self-attention model.
            multi_head_attention_activation_list:   `list` of `str` of activation function for multi-head attention model.
            fc_activation_list:             `list` of `str` of activation function in fully-connected layers.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.

        """
        if isinstance(depth_dim, int) is False:
            raise TypeError("The type of `depth_dim` must be `int`.")
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(TransformerDecoder, self).__init__()
        self.__computable_loss = computable_loss
        self.optimizer_f = optimizer_f
        self.optimizer = None
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__not_init_flag = not_init_flag

        self.layer_n = layer_n

        if len(self_attention_activation_list) != layer_n:
            self_attention_activation_list = self_attention_activation_list * layer_n
        self.self_attention_activation_list = nn.ModuleList(
            self_attention_activation_list
        )

        if len(multi_head_attention_activation_list) != layer_n:
            multi_head_attention_activation_list = multi_head_attention_activation_list * layer_n
        self.multi_head_attention_activation_list = nn.ModuleList(
            multi_head_attention_activation_list
        )

        if len(fc_activation_list) != layer_n:
            fc_activation_list = fc_activation_list * layer_n
        self.fc_activation_list = nn.ModuleList(
            fc_activation_list
        )

        self.embedding_dropout = nn.Dropout(p=dropout_rate)

        if len(self_attention_layer_norm_list) == 0:
            self.self_attention_layer_norm_list = [None] * layer_n
        elif self_attention_layer_norm_list[0] == "auto":
            self.self_attention_layer_norm_list = ["auto"] * layer_n
        else:
            self.self_attention_layer_norm_list = nn.ModuleList(
                self_attention_layer_norm_list
            )

        if len(multi_head_attention_layer_norm_list) == 0:
            self.multi_head_attention_layer_norm_list = [None] * layer_n
        elif multi_head_attention_layer_norm_list[0] == "auto":
            self.multi_head_attention_layer_norm_list = ["auto"] * layer_n
        else:
            self.multi_head_attention_layer_norm_list = nn.ModuleList(
                multi_head_attention_layer_norm_list
            )

        self_attention_list = [None] * layer_n
        multi_head_attention_list = [None] * layer_n
        multi_head_attention_dropout_list = [None] * layer_n
        self_attention_dropout_list = [None] * layer_n
        fc_dropout_list = [None] * layer_n

        if len(filter_fc_list) == 0:
            self.filter_fc_list = [None] * layer_n
        else:
            self.filter_fc_list = nn.ModuleList(
                filter_fc_list
            )

        if len(fc_layer_norm_list) == 0:
            self.fc_layer_norm_list = [None] * layer_n
        elif fc_layer_norm_list[0] == "auto":
            self.fc_layer_norm_list = ["auto"] * layer_n
        else:
            self.fc_layer_norm_list = fc_layer_norm_list

        if len(fc_list) == 0:
            self.fc_list = [None] * layer_n
        else:
            self.fc_list = nn.ModuleList(fc_list)

        for i in range(layer_n):
            self_attention_list[i] = SelfAttentionModel(
                depth_dim=depth_dim,
                computable_loss=computable_loss,
                not_init_flag=not_init_flag,
                ctx=ctx,
            )
            self_attention_list[i].head_n = head_n
            self_attention_dropout_list[i] = nn.Dropout(p=dropout_rate)
            multi_head_attention_list[i] = MultiHeadAttentionModel(
                depth_dim=depth_dim,
                computable_loss=computable_loss,
                not_init_flag=not_init_flag,
                ctx=ctx,
            )
            multi_head_attention_list[i].head_n = head_n
            multi_head_attention_dropout_list[i] = nn.Dropout(p=dropout_rate)
            fc_dropout_list[i] = nn.Dropout(p=dropout_rate)

        self.self_attention_list = nn.ModuleList(self_attention_list)
        self.self_attention_dropout_list = nn.ModuleList(self_attention_dropout_list)
        self.multi_head_attention_list = nn.ModuleList(multi_head_attention_list)
        self.multi_head_attention_dropout_list = nn.ModuleList(multi_head_attention_dropout_list)
        self.fc_dropout_list = nn.ModuleList(fc_dropout_list)

        self.output_layer_norm = output_layer_norm
        self.output_fc = output_fc

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx
        self.to(self.__ctx)

        logger = getLogger("accelbrainbase")
        self.logger = logger

        self.head_n = head_n
        self.depth_dim = depth_dim

    def learn(self, iteratable_data):
        '''
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

        Args:
            iteratable_data:     is-a `IteratableData`.

        '''
        raise NotImplementedError("This class itself can do learning. Use `TransformerController`.")

    def inference(
        self, 
        observed_arr, 
        encoded_arr, 
        self_attention_mask_arr=None, 
        multi_head_attention_mask_arr=None
    ):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:                   rank-3 Array like or sparse matrix as the observed data points.
                                            The shape is: (batch size, the length of sequence, feature points)

            encoded_arr:                    encoded array by `TransformerEncoder`.

            self_attention_mask_arr:       rank-3 Array like or sparse matrix as the mask.
                                           The shape is: (batch size, the length of sequence, feature points)

            multi_head_attention_mask_arr  rank-3 Array like or sparse matrix as the mask.
                                           The shape is: (batch size, the length of sequence, feature points)
 
        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(
            observed_arr, 
            encoded_arr, 
            self_attention_mask_arr, 
            multi_head_attention_mask_arr
        )

    def forward(
        self, 
        observed_arr, 
        encoded_arr, 
        self_attention_mask_arr=None, 
        multi_head_attention_mask_arr=None
    ):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:                              `mxnet.ndarray` or `mxnet.symbol`.
            observed_arr:                   rank-3 Array like or sparse matrix as the observed data points.
                                            The shape is: (batch size, the length of sequence, feature points)

            encoded_arr:                    encoded array by `TransformerEncoder`.

            self_attention_mask_arr:       rank-3 Array like or sparse matrix as the mask.
                                           The shape is: (batch size, the length of sequence, feature points)

            multi_head_attention_mask_arr  rank-3 Array like or sparse matrix as the mask.
                                           The shape is: (batch size, the length of sequence, feature points)
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        observed_arr = self.embedding(observed_arr)

        if self_attention_mask_arr is None:
            self_attention_mask_arr = torch.ones(
                (
                    observed_arr.shape[0],
                    1,
                    1
                )
            )
            self_attention_mask_arr = self_attention_mask_arr.to(observed_arr.device)
        if multi_head_attention_mask_arr is None:
            multi_head_attention_mask_arr = torch.ones(
                shape=(
                    observed_arr.shape[0],
                    1,
                    1
                ),
            )
            multi_head_attention_mask_arr = multi_head_attention_mask_arr.to(observed_arr.device)

        if self.embedding_flag is True:
            x = self.embedding_dropout(observed_arr)
        else:
            x = observed_arr

        for i in range(self.layer_n):
            if self.self_attention_layer_norm_list[i] is None:
                self.self_attention_layer_norm_list[i] = nn.LayerNorm(
                    x.shape[-1],
                    device=x.device
                )
                self.self_attention_layer_norm_list[i].to(self.__ctx)
            _x = self.self_attention_layer_norm_list[i](x)
            _x = self.self_attention_list[i](_x, self_attention_mask_arr)
            _x = self.self_attention_dropout_list[i](_x)

            if len(self.self_attention_activation_list) > 0:
                if self.self_attention_activation_list[i] is not None:
                    _x = self.self_attention_activation_list[i](_x)

            x = x + _x

            if self.multi_head_attention_layer_norm_list[i] is None:
                self.multi_head_attention_layer_norm_list[i] = nn.LayerNorm(
                    x.shape[-1],
                    device=x.device
                )
                self.multi_head_attention_layer_norm_list[i].to(self.__ctx)

            _x = self.multi_head_attention_layer_norm_list[i](x)
            _x = self.multi_head_attention_list[i](_x, encoded_arr, multi_head_attention_mask_arr)
            _x = self.multi_head_attention_dropout_list[i](_x)

            if len(self.multi_head_attention_activation_list) > 0:
                if self.multi_head_attention_activation_list[i] is not None:
                    _x = self.multi_head_attention_activation_list[i](_x)

            x = x + _x

            if self.fc_layer_norm_list[i] is None:
                self.fc_layer_norm_list[i] = nn.LayerNorm(
                    x.shape[-1],
                    device=x.device
                )
                self.fc_layer_norm_list[i].to(self.__ctx)
            _x = self.fc_layer_norm_list[i](x)

            if self.filter_fc_list[i] is None:
                self.filter_fc_list[i] = nn.Linear(
                    _x.shape[-1],
                    _x.shape[-1]
                )
                self.filter_fc_list[i].to(self.__ctx)
            _x = self.filter_fc_list[i](_x)
            _x = self.fc_dropout_list[i](_x)

            if self.fc_list[i] is None:
                self.fc_list[i] = nn.Linear(
                    _x.shape[-1],
                    _x.shape[-1]
                )
                self.fc_list[i].to(self.__ctx)
            _x = self.fc_list[i](_x)

            if len(self.fc_activation_list) > 0:
                if self.fc_activation_list[i] is not None:
                    _x = self.fc_activation_list[i](_x)

            x = x + _x

        if self.output_layer_norm is None:
            self.output_layer_norm = nn.LayerNorm(
                x.shape[-1],
                device=x.device
            )
            self.output_layer_norm.to(self.__ctx)
        y = self.output_layer_norm(x)

        if self.output_fc is None:
            self.output_fc = nn.Linear(
                y.shape[-1],
                y.shape[-1]
            )
            self.output_fc.to(self.__ctx)
        y = self.output_fc(y)

        if self.optimizer is None:
            if self.__not_init_flag is False:
                if self.optimizer_f is not None:
                    self.optimizer = self.optimizer_f(
                        self.parameters()
                    )
                else:
                    self.optimizer = AdamW(
                        self.parameters(),
                        lr=self.__learning_rate,
                        weight_decay=self.__weight_decay
                    )

        return y
