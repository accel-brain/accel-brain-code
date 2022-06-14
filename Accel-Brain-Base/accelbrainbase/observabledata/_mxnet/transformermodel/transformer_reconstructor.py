# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.transformer_model import TransformerModel
from accelbrainbase.observabledata._mxnet.attention_model import AttentionModel
from accelbrainbase.observabledata._mxnet.attentionmodel.multi_head_attention_model import MultiHeadAttentionModel
from accelbrainbase.observabledata._mxnet.attentionmodel.multiheadattentionmodel.self_attention_model import SelfAttentionModel
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase.regularizatable_data import RegularizatableData

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import MXNetError
from logging import getLogger


class TransformerReconstructor(HybridBlock, TransformerModel):
    '''
    Reconstructor of Transformering Auto-Encoder.

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
            self.self_attention_list[i].batch_size = value

    batch_size = property(get_batch_size, set_batch_size)

    __head_n = None

    def get_head_n(self):
        ''' getter '''
        return self.__head_n
    
    def set_head_n(self, value):
        ''' setter '''
        self.__head_n = value
        for i in range(len(self.self_attention_list)):
            self.self_attention_list[i].head_n = value

    head_n = property(get_head_n, set_head_n)

    __seq_len = None

    def get_seq_len(self):
        ''' getter '''
        return self.__seq_len
    
    def set_seq_len(self, value):
        ''' setter '''
        self.__seq_len = value
        for i in range(len(self.self_attention_list)):
            self.self_attention_list[i].seq_len = value

    seq_len = property(get_seq_len, set_seq_len)

    __depth_dim = None

    def get_depth_dim(self):
        ''' getter '''
        return self.__depth_dim
    
    def set_depth_dim(self, value):
        ''' setter '''
        self.__depth_dim = value
        for i in range(len(self.self_attention_list)):
            self.self_attention_list[i].depth_dim = value

    depth_dim = property(get_depth_dim, set_depth_dim)

    def __init__(
        self,
        depth_dim,
        layer_n,
        computable_loss,
        head_n=3,
        initializer=None,
        self_attention_layer_norm_list=[],
        fc_layer_norm_list=[],
        filter_fc_list=[],
        dropout_rate=0.1,
        fc_list=[],
        self_attention_activation_list=[],
        fc_activation_list=[],
        output_layer_norm=None,
        output_fc=None,
        output_dim=None,
        optimizer_name="SGD",
        learning_rate=1e-03,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        regularizatable_data_list=[],
        ctx=mx.gpu(),
        hybridize_flag=True,
        not_init_flag=False,
        **kwargs
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
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, gluon.loss.Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(TransformerReconstructor, self).__init__()
        self.__computable_loss = computable_loss

        if initializer is None:
            self.initializer = mx.initializer.Normal(1.0)
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")

            self.initializer = initializer

        self.layer_n = layer_n

        self.self_attention_activation_list = self_attention_activation_list
        if len(self.self_attention_activation_list) != layer_n:
            self.self_attention_activation_list = self.self_attention_activation_list * layer_n

        self.fc_activation_list = fc_activation_list
        if len(self.fc_activation_list) != layer_n:
            self.fc_activation_list = self.fc_activation_list * layer_n

        with self.name_scope():
            self.embedding_dropout = gluon.nn.Dropout(rate=dropout_rate)

            if len(self_attention_layer_norm_list) == 0:
                self.self_attention_layer_norm_list = [None] * layer_n
            else:
                self.self_attention_layer_norm_list = self_attention_layer_norm_list

            self.self_attention_list = [None] * layer_n

            if len(filter_fc_list) == 0:
                self.filter_fc_list = [None] * layer_n
            else:
                self.filter_fc_list = filter_fc_list

            self.self_attention_dropout_list = [None] * layer_n

            if len(fc_layer_norm_list) == 0:
                self.fc_layer_norm_list = [None] * layer_n
            else:
                self.fc_layer_norm_list = fc_layer_norm_list

            self.fc_dropout_list = [None] * layer_n

            if len(fc_list) == 0:
                self.fc_list = [None] * layer_n
            else:
                self.fc_list = fc_list

            for i in range(layer_n):
                if self.self_attention_layer_norm_list[i] is None:
                    self.self_attention_layer_norm_list[i] = gluon.nn.LayerNorm()
                self.register_child(self.self_attention_layer_norm_list[i])
                self.self_attention_list[i] = SelfAttentionModel(
                    depth_dim=depth_dim,
                    computable_loss=computable_loss,
                    not_init_flag=not_init_flag,
                )
                self.self_attention_list[i].head_n = head_n
                self.register_child(self.self_attention_list[i])

                self.self_attention_dropout_list[i] = gluon.nn.Dropout(rate=dropout_rate)
                self.register_child(self.self_attention_dropout_list[i])

                if self.fc_layer_norm_list[i] is None:
                    self.fc_layer_norm_list[i] = gluon.nn.LayerNorm()
                self.register_child(self.fc_layer_norm_list[i])

                if self.filter_fc_list[i] is None:
                    self.filter_fc_list[i] = gluon.nn.Dense(
                        depth_dim, 
                        use_bias=True,
                        flatten=False,
                    )
                self.register_child(self.filter_fc_list[i])

                if self.fc_list[i] is None:
                    self.fc_list[i] = gluon.nn.Dense(
                        depth_dim, 
                        use_bias=True,
                        flatten=False,
                    )
                self.register_child(self.fc_list[i])

                self.fc_dropout_list[i] = gluon.nn.Dropout(rate=dropout_rate)
                self.register_child(self.fc_dropout_list[i])

                if self.self_attention_activation_list[i] == "GELU" or self.self_attention_activation_list[i] == "gelu":
                    self.self_attention_activation_list[i] = mx.gluon.nn.GELU()
                    self.register_child(self.self_attention_activation_list[i])

                if self.fc_activation_list[i] == "GELU" or self.fc_activation_list[i] == "gelu":
                    self.fc_activation_list[i] = mx.gluon.nn.GELU()
                    self.register_child(self.fc_activation_list[i])

            if output_layer_norm is None:
                self.output_layer_norm = gluon.nn.LayerNorm()
            else:
                self.output_layer_norm = output_layer_norm
            self.register_child(self.output_layer_norm)

            if output_fc is None:
                if output_dim is None:
                    output_dim = depth_dim
                self.output_fc = gluon.nn.Dense(
                    output_dim,
                    use_bias=False,
                    flatten=False,
                )
            else:
                self.output_fc = output_fc
            self.register_child(self.output_fc)

        if self.init_deferred_flag is False:
            if not_init_flag is False:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(self.collect_params(), optimizer_name, {"learning_rate": learning_rate})
                if hybridize_flag is True:
                    self.hybridize()

        self.learning_rate = learning_rate
        self.learning_attenuate_rate = learning_attenuate_rate
        self.attenuate_epoch = attenuate_epoch

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx

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
        mask=None, 
    ):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:                   rank-3 Array like or sparse matrix as the observed data points.
                                            The shape is: (batch size, the length of sequence, feature points)

            mask:       rank-3 Array like or sparse matrix as the mask.
                                           The shape is: (batch size, the length of sequence, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        observed_arr = self.embedding(observed_arr)

        if mask is None:
            mask = nd.ones(
                shape=(
                    observed_arr.shape[0],
                    1,
                    1,
                    1
                ),
                ctx=observed_arr.context
            )

        return self(
            observed_arr, 
            mask, 
        )

    def hybrid_forward(
        self, 
        F,
        observed_arr, 
        mask=None, 
    ):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:                              `mxnet.ndarray` or `mxnet.symbol`.
            observed_arr:                   rank-3 Array like or sparse matrix as the observed data points.
                                            The shape is: (batch size, the length of sequence, feature points)

            mask:       rank-3 Array like or sparse matrix as the mask.
                                           The shape is: (batch size, the length of sequence, feature points)
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(
            F, 
            observed_arr, 
            mask, 
        )

    def forward_propagation(
        self, 
        F,
        observed_arr, 
        mask=None, 
    ):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:                              `mxnet.ndarray` or `mxnet.symbol`.
            observed_arr:                   rank-3 Array like or sparse matrix as the observed data points.
                                            The shape is: (batch size, the length of sequence, feature points)

            mask:       rank-3 Array like or sparse matrix as the mask.
                                           The shape is: (batch size, the length of sequence, feature points)
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        if self.embedding_flag is True:
            x = self.embedding_dropout(observed_arr)
        else:
            x = observed_arr

        for i in range(self.layer_n):
            _x = self.self_attention_layer_norm_list[i](x)
            _x = self.self_attention_list[i].inference(_x, mask)
            _x = self.self_attention_dropout_list[i](_x)

            if len(self.self_attention_activation_list) > 0:
                if isinstance(self.self_attention_activation_list[i], gluon.nn.activations.GELU) is True:
                    _x = self.self_attention_activation_list[i](_x)
                elif self.self_attention_activation_list[i] == "identity_adjusted":
                    _x = _x / F.sum(F.ones_like(_x))
                elif self.self_attention_activation_list[i] == "softmax":
                    _x = F.softmax(_x)
                elif self.self_attention_activation_list[i] == "log_softmax":
                    _x = F.softmax(_x)
                elif self.self_attention_activation_list[i] != "identity":
                    _x = F.Activation(_x, self.self_attention_activation_list[i])

            x = x + _x

            _x = self.fc_layer_norm_list[i](x)
            _x = self.filter_fc_list[i](_x)
            _x = self.fc_dropout_list[i](_x)
            _x = self.fc_list[i](_x)

            if len(self.fc_activation_list) > 0:
                if isinstance(self.fc_activation_list[i], gluon.nn.activations.GELU) is True:
                    _x = self.fc_activation_list[i](_x)
                elif self.fc_activation_list[i] == "identity_adjusted":
                    _x = _x / F.sum(F.ones_like(_x))
                elif self.fc_activation_list[i] == "softmax":
                    _x = F.softmax(_x)
                elif self.fc_activation_list[i] == "log_softmax":
                    _x = F.softmax(_x)
                elif self.fc_activation_list[i] != "identity":
                    _x = F.Activation(_x, self.fc_activation_list[i])

            x = x + _x

        y = self.output_layer_norm(x)
        y = self.output_fc(y)

        return y
