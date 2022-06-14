# -*- coding: utf-8 -*-
from accelbrainbase.controllable_model import ControllableModel
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError
from accelbrainbase.observabledata._mxnet.transformer_model import TransformerModel
from accelbrainbase.observabledata._mxnet.transformermodel.transformer_encoder import TransformerEncoder
from accelbrainbase.observabledata._mxnet.transformermodel.transformer_decoder import TransformerDecoder
from accelbrainbase.iteratabledata.transformer_iterator import TransformerIterator
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


class TransformerController(HybridBlock, ControllableModel):
    '''
    Transformer.

    References:
        - Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
        - Floridi, L., & Chiriatti, M. (2020). GPT-3: Its nature, scope, limits, and consequences. Minds and Machines, 30(4), 681-694.
        - Miller, A., Fisch, A., Dodge, J., Karimi, A. H., Bordes, A., & Weston, J. (2016). Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126.
        - Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018) Improving Language Understanding by Generative Pre-Training. OpenAI (URL: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
        - Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.

    '''
    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        computable_loss=None,
        encoder=None,
        decoder=None,
        layer_n=3,
        head_n=3,
        seq_len=5,
        depth_dim=100,
        hidden_dim=100,
        self_attention_activation_list=[],
        multi_head_attention_activation_list=[],
        fc_activation_list=[],
        optimizer_name="SGD",
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        dropout_rate=0.5,
        hybridize_flag=True,
        ctx=mx.gpu(),
        initializer=None,
        regularizatable_data_list=[],
        weight_decay=0.01,
        positional_embedding_weignt=1.0,
        **kwargs
    ):
        '''
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

        '''
        super(TransformerController, self).__init__()

        if computable_loss is None:
            computable_loss = gluon.loss.SoftmaxCrossEntropyLoss(
                axis=-1, 
                sparse_label=False, 
                from_logits=False, 
                weight=None, 
                batch_axis=0
            )
        self.__computable_loss = computable_loss

        if encoder is None:
            if hidden_dim is None or hidden_dim == depth_dim:
                encoder = TransformerEncoder(
                    depth_dim=depth_dim,
                    layer_n=layer_n,
                    head_n=head_n,
                    self_attention_activation_list=self_attention_activation_list,
                    fc_activation_list=fc_activation_list,
                    computable_loss=computable_loss,
                    initializer=initializer,
                    not_init_flag=True,
                    hybridize_flag=hybridize_flag,
                    dropout_rate=dropout_rate
                )
            else:
                encoder = TransformerEncoder(
                    depth_dim=hidden_dim,
                    layer_n=layer_n,
                    head_n=head_n,
                    self_attention_activation_list=self_attention_activation_list,
                    fc_activation_list=fc_activation_list,
                    computable_loss=computable_loss,
                    initializer=initializer,
                    not_init_flag=True,
                    hybridize_flag=hybridize_flag,
                    dropout_rate=dropout_rate
                )
            encoder.embedding_weignt = positional_embedding_weignt
        else:
            if isinstance(encoder, TransformerModel) is False:
                raise TypeError("The type of `encoder` must be `TransformerModel`.")

        if decoder is None:
            if hidden_dim is None or hidden_dim == depth_dim:
                decoder = TransformerDecoder(
                    head_n=head_n,
                    depth_dim=depth_dim,
                    layer_n=layer_n,
                    self_attention_activation_list=self_attention_activation_list,
                    multi_head_attention_activation_list=multi_head_attention_activation_list,
                    fc_activation_list=fc_activation_list,
                    computable_loss=computable_loss,
                    initializer=initializer,
                    not_init_flag=True,
                    hybridize_flag=hybridize_flag
                )
            else:
                decoder = TransformerDecoder(
                    head_n=head_n,
                    depth_dim=hidden_dim,
                    output_dim=depth_dim,
                    layer_n=layer_n,
                    self_attention_activation_list=self_attention_activation_list,
                    multi_head_attention_activation_list=multi_head_attention_activation_list,
                    fc_activation_list=fc_activation_list,
                    computable_loss=computable_loss,
                    initializer=initializer,
                    not_init_flag=True,
                    hybridize_flag=hybridize_flag
                )
            decoder.embedding_weignt = positional_embedding_weignt
        else:
            if isinstance(decoder, TransformerModel) is False:
                raise TypeError("The type of `decoder` must be `TransformerModel`.")

        logger = getLogger("accelbrainbase")
        self.logger = logger

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=2
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")
            self.initializer = initializer

        self.encoder = encoder
        self.decoder = decoder

        with self.name_scope():
            if hidden_dim is not None and hidden_dim != depth_dim:
                self.encoder_hidden_fc = gluon.nn.Dense(
                    hidden_dim, 
                    use_bias=True,
                    flatten=False,
                )
                self.register_child(self.encoder_hidden_fc)
                self.decoder_hidden_fc = gluon.nn.Dense(
                    hidden_dim, 
                    use_bias=True,
                    flatten=False,
                )
                self.register_child(self.decoder_hidden_fc)
            else:
                self.encoder_hidden_fc = None
                self.decoder_hidden_fc = None

            self.register_child(self.encoder)
            self.register_child(self.decoder)

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(
                    self.collect_params(), 
                    optimizer_name, 
                    {
                        "learning_rate": learning_rate,
                        "wd": weight_decay
                    }
                )
                if hybridize_flag is True:
                    self.encoder.hybridize()
                    self.decoder.hybridize()

            except InitDeferredError:
                self.logger.debug("The initialization should be deferred.")

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.seq_len = seq_len

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.encoder.collect_params(select)
        params_dict.update(self.decoder.collect_params(select))
        if self.encoder_hidden_fc is not None:
            params_dict.update(self.encoder_hidden_fc.collect_params(select))
        if self.decoder_hidden_fc is not None:
            params_dict.update(self.decoder_hidden_fc.collect_params(select))
        return params_dict

    def learn(self, iteratable_data):
        '''
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

        Args:
            iteratable_data:     is-a `TransformerIterator`.
        '''
        if isinstance(iteratable_data, TransformerIterator) is False:
            raise TypeError("The type of `iteratable_data` must be `TransformerIterator`.")

        self.__loss_list = []
        learning_rate = self.__learning_rate

        try:
            epoch = 0
            iter_n = 0
            for encoded_observed_arr, decoded_observed_arr, encoded_mask_arr, decoded_mask_arr, test_encoded_observed_arr, test_decoded_observed_arr, test_encoded_mask_arr, test_decoded_mask_arr, training_target_arr, test_target_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.trainer.set_learning_rate(learning_rate)

                with autograd.record():
                    pred_arr = self.inference(
                        encoded_observed_arr, 
                        decoded_observed_arr, 
                        encoded_mask_arr, 
                        decoded_mask_arr
                    )
                    loss = self.compute_loss(
                        pred_arr,
                        training_target_arr
                    )
                loss.backward()
                self.trainer.step(encoded_observed_arr.shape[0])
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    test_pred_arr = self.inference(
                        test_encoded_observed_arr, 
                        test_decoded_observed_arr, 
                        test_encoded_mask_arr, 
                        test_decoded_mask_arr
                    )

                    test_loss = self.compute_loss(
                        test_pred_arr,
                        test_target_arr
                    )
                    self.__loss_list.append((loss.asnumpy().mean(), test_loss.asnumpy().mean()))
                    self.logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(loss.asnumpy().mean()) + " Test loss: " + str(test_loss.asnumpy().mean()))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.logger.debug("Interrupt.")

        self.logger.debug("end. ")

    def inference(
        self, 
        encoded_observed_arr, 
        decoded_observed_arr, 
        encoded_mask_arr=None,
        decoded_mask_arr=None,
    ):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            encoded_observed_arr:   rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            decoded_observed_arr:   rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            encoded_mask_arr:       rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            decoded_mask_arr:       rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        if encoded_mask_arr is None:
            encoded_mask_arr = nd.ones(
                shape=(encoded_observed_arr.shape[0], 1, 1, 1), 
                ctx=encoded_observed_arr.context
            )
        if decoded_mask_arr is None:
            decoded_mask_arr = nd.ones(
                shape=(decoded_observed_arr.shape[0], 1, 1, 1), 
                ctx=decoded_observed_arr.context
            )

        return self(
            encoded_observed_arr, 
            decoded_observed_arr, 
            encoded_mask_arr,
            decoded_mask_arr,
        )

    def compute_loss(self, pred_arr, labeled_arr):
        '''
        Compute loss.

        Args:
            pred_arr:       `mxnet.ndarray` or `mxnet.symbol`.
            labeled_arr:    `mxnet.ndarray` or `mxnet.symbol`.

        Returns:
            loss.
        '''
        return self.__computable_loss(pred_arr, labeled_arr)

    def hybrid_forward(
        self, 
        F, 
        encoded_observed_arr, 
        decoded_observed_arr, 
        encoded_mask_arr=None,
        decoded_mask_arr=None,
    ):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            encoded_observed_arr:   rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            decoded_observed_arr:   rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            encoded_mask_arr:       rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            decoded_mask_arr:       rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(
            F, 
            encoded_observed_arr, 
            decoded_observed_arr, 
            encoded_mask_arr,
            decoded_mask_arr,
        )

    def forward_propagation(
        self, 
        F, 
        encoded_observed_arr, 
        decoded_observed_arr, 
        encoded_mask_arr=None,
        decoded_mask_arr=None,
    ):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            encoded_observed_arr:   rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            decoded_observed_arr:   rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            encoded_mask_arr:       rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)

            decoded_mask_arr:       rank-3 Array like or sparse matrix as the observed data points.
                                    The shape is: (batch size, the length of sequence, feature points)
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        if self.encoder_hidden_fc is not None:
            encoded_observed_arr = self.encoder_hidden_fc(encoded_observed_arr)
        if self.decoder_hidden_fc is not None:
            decoded_observed_arr = self.decoder_hidden_fc(decoded_observed_arr)

        steps_arr = F.arange(self.seq_len)
        mask_arr = F.broadcast_lesser_equal(
            steps_arr.reshape((1, -1)),
            steps_arr.reshape((-1, 1))
        )
        ones_arr = F.ones_like(steps_arr)
        seq_len_arr = ones_arr * self.seq_len
        batch_mask_arr = F.broadcast_lesser(
            steps_arr.reshape((1, -1)),
            seq_len_arr.reshape((-1, 1))
        )
        _decoded_mask_arr = F.broadcast_mul(batch_mask_arr, F.expand_dims(mask_arr, 0))
        _decoded_mask_arr = F.expand_dims(_decoded_mask_arr, 0)
        _decoded_mask_arr = _decoded_mask_arr + 1e-08

        if decoded_mask_arr is None:
            decoded_mask_arr = _decoded_mask_arr
        else:
            decoded_mask_arr = F.broadcast_add(decoded_mask_arr, _decoded_mask_arr)

        encoded_arr = self.encoder.inference(
            encoded_observed_arr,
            encoded_mask_arr
        )
        self.feature_points_arr = encoded_arr
        decoded_arr = self.decoder.inference(
            decoded_observed_arr, 
            encoded_arr, 
            decoded_mask_arr,
            encoded_mask_arr,
        )
        return decoded_arr

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = self.collect_params()
        
        params_arr_dict = {}
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        return params_arr_dict

    def regularize(self):
        '''
        Regularization.
        '''
        params_dict = self.extract_learned_dict()
        for regularizatable in self.__regularizatable_data_list:
            params_dict = regularizatable.regularize(params_dict)

        for k, params in self.collect_params().items():
            params.set_data(params_dict[k])

    def __rename_file(self, filename):
        filename_list = filename.split(".")
        _format = filename_list[-1]
        g_filename = filename.replace("." + _format, "_encoder." + _format)
        d_filename = filename.replace("." + _format, "_decoder." + _format)
        return g_filename, d_filename

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        e_filename, d_filename = self.__rename_file(filename)
        self.encoder.save_parameters(e_filename)
        self.decoder.save_parameters(d_filename)

    def load_parameters(self, filename, ctx=None, allow_missing=False, ignore_extra=False):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            `mx.cpu()` or `mx.gpu()`.
            allow_missing:  `bool` of whether to silently skip loading parameters not represents in the file.
            ignore_extra:   `bool` of whether to silently ignre parameters from the file that are not present in this `Block`.
        '''
        e_filename, d_filename = self.__rename_file(filename)
        self.encoder.load_parameters(e_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        self.decoder.load_parameters(d_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not.'''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    def get_loss_arr(self):
        ''' getter '''
        return np.array(self.__loss_list)
    
    def set_loss_arr(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    loss_arr = property(get_loss_arr, set_loss_arr)
