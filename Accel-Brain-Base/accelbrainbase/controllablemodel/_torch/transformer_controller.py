# -*- coding: utf-8 -*-
from accelbrainbase.controllable_model import ControllableModel
from accelbrainbase.observabledata._torch.transformer_model import TransformerModel
from accelbrainbase.observabledata._torch.transformermodel.transformer_encoder import TransformerEncoder
from accelbrainbase.observabledata._torch.transformermodel.transformer_decoder import TransformerDecoder
from accelbrainbase.iteratabledata.transformer_iterator import TransformerIterator
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase.regularizatable_data import RegularizatableData
import numpy as np
from logging import getLogger
import torch
from torch import nn
from torch.optim.adamw import AdamW


class TransformerController(nn.Module, ControllableModel):
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

    __loaded_filename = None
    __loaded_ctx = None

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        computable_loss=None,
        encoder=None,
        decoder=None,
        output_nn=None,
        optimizer_f=None,
        layer_n=3,
        head_n=3,
        seq_len=5,
        depth_dim=100,
        hidden_dim=100,
        self_attention_activation_list=[
            torch.nn.GELU(),
            torch.nn.GELU(),
            torch.nn.GELU(),
        ],
        multi_head_attention_activation_list=[
            torch.nn.GELU(),
            torch.nn.GELU(),
            torch.nn.GELU(),
        ],
        fc_activation_list=[
            torch.nn.GELU(),
            torch.nn.GELU(),
            torch.nn.GELU(),
        ],
        learning_rate=6e-06,
        weight_decay=0.01,
        dropout_rate=0.5,
        ctx="cpu",
        regularizatable_data_list=[],
        positional_embedding_weignt=1.0,
        not_init_flag=False,
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
            computable_loss = nn.CrossEntropyLoss()

        self.__computable_loss = computable_loss
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__not_init_flag = not_init_flag
        self.__ctx = ctx

        if encoder is None:
            if hidden_dim is None or hidden_dim == depth_dim:
                encoder = TransformerEncoder(
                    depth_dim=depth_dim,
                    layer_n=layer_n,
                    head_n=head_n,
                    self_attention_activation_list=self_attention_activation_list,
                    fc_activation_list=fc_activation_list,
                    computable_loss=computable_loss,
                    not_init_flag=not_init_flag,
                    dropout_rate=dropout_rate,
                    ctx=ctx,
                )
            else:
                encoder = TransformerEncoder(
                    depth_dim=hidden_dim,
                    layer_n=layer_n,
                    head_n=head_n,
                    self_attention_activation_list=self_attention_activation_list,
                    fc_activation_list=fc_activation_list,
                    computable_loss=computable_loss,
                    not_init_flag=not_init_flag,
                    dropout_rate=dropout_rate,
                    ctx=ctx,
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
                    not_init_flag=not_init_flag,
                    ctx=ctx,
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
                    not_init_flag=not_init_flag,
                    ctx=ctx,
                )
            decoder.embedding_weignt = positional_embedding_weignt
        else:
            if isinstance(decoder, TransformerModel) is False:
                raise TypeError("The type of `decoder` must be `TransformerModel`.")

        logger = getLogger("accelbrainbase")
        self.logger = logger

        self.encoder = encoder
        self.decoder = decoder
        self.output_nn = output_nn

        if hidden_dim is not None and hidden_dim != depth_dim:
            self.encoder_hidden_fc = nn.Linear(
                depth_dim,
                hidden_dim, 
                bias=True,
            )
            self.decoder_hidden_fc = nn.Linear(
                depth_dim,
                hidden_dim, 
                bias=True,
            )
        else:
            self.encoder_hidden_fc = None
            self.decoder_hidden_fc = None

        self.__ctx = ctx
        self.to(self.__ctx)

        if self.init_deferred_flag is False:
            if not_init_flag is False:
                if optimizer_f is not None:
                    self.optimizer = optimizer_f(
                        self.parameters(),
                    )
                else:
                    self.optimizer = AdamW(
                        self.parameters(),
                        lr=self.__learning_rate,
                        weight_decay=weight_decay
                    )

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.seq_len = seq_len
        self.epoch = 0

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
            epoch = self.epoch
            iter_n = 0
            for encoded_observed_arr, decoded_observed_arr, encoded_mask_arr, decoded_mask_arr, test_encoded_observed_arr, test_decoded_observed_arr, test_encoded_mask_arr, test_decoded_mask_arr, training_target_arr, test_target_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                if self.encoder.optimizer is not None and self.decoder.optimizer is not None:
                    optimizer_setup_flag = True
                    self.encoder.optimizer.zero_grad()
                    self.decoder.optimizer.zero_grad()
                    self.optimizer.zero_grad()
                    if self.output_nn is not None:
                        self.output_nn.optimizer.zero_grad()
                else:
                    optimizer_setup_flag = False

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
                if optimizer_setup_flag is False:
                    self.encoder.optimizer.zero_grad()
                    self.decoder.optimizer.zero_grad()
                    self.optimizer.zero_grad()
                    if self.output_nn is not None:
                        self.output_nn.optimizer.zero_grad()

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
                self.optimizer.step()
                if self.output_nn is not None:
                    self.output_nn.optimizer.step()

                self.decoder.optimizer.step()
                self.encoder.optimizer.step()
                self.regularize()
                self.decoder.regularize()
                self.encoder.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    if torch.inference_mode():
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
                    _loss = loss.to('cpu').detach().numpy().copy()
                    _test_loss = test_loss.to('cpu').detach().numpy().copy()

                    self.__loss_list.append((_loss, _test_loss))
                    self.logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(_loss) + " Test loss: " + str(_test_loss))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.logger.debug("Interrupt.")

        self.logger.debug("end. ")
        self.epoch = epoch

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

    def forward(
        self, 
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
        if self.__loaded_filename is not None:
            loaded_filename = self.__loaded_filename
            self.__loaded_filename = None
            init_encoded_observed_arr = encoded_observed_arr.detach()
            init_decoded_observed_arr = decoded_observed_arr.detach()
            if encoded_mask_arr is not None:
                init_encoded_mask_arr = encoded_mask_arr.detach()
            else:
                init_encoded_mask_arr = None
            if decoded_mask_arr is not None:
                init_decoded_mask_arr = decoded_mask_arr.detach()
            else:
                init_decoded_mask_arr = decoded_mask_arr

            _ = self.forward(
                init_encoded_observed_arr, 
                init_decoded_observed_arr, 
                init_encoded_mask_arr,
                init_decoded_mask_arr,
            )
            self.load_parameters(loaded_filename, ctx=self.__loaded_ctx)
            self.__loaded_ctx = None

        if decoded_mask_arr is not None:
            init_decoded_mask_arr = decoded_mask_arr.detach()
        else:
            init_decoded_mask_arr = None

        if encoded_mask_arr is None:
            encoded_mask_arr = torch.ones(
                (encoded_observed_arr.shape[0], 1, 1, 1), 
            )
            encoded_mask_arr = encoded_mask_arr.to(encoded_observed_arr.device)
        if decoded_mask_arr is None:
            decoded_mask_arr = torch.ones(
                (decoded_observed_arr.shape[0], 1, 1, 1), 
            )
            decoded_mask_arr = decoded_mask_arr.to(decoded_observed_arr.device)

        if self.encoder_hidden_fc is not None:
            encoded_observed_arr = self.encoder_hidden_fc(encoded_observed_arr)
        if self.decoder_hidden_fc is not None:
            decoded_observed_arr = self.decoder_hidden_fc(decoded_observed_arr)

        steps_arr = torch.arange(self.seq_len, device=encoded_observed_arr.device)
        mask_arr = torch.le(
            steps_arr.reshape((1, -1)),
            steps_arr.reshape((-1, 1))
        )
        ones_arr = torch.ones_like(steps_arr)
        seq_len_arr = ones_arr * self.seq_len
        batch_mask_arr = torch.lt(
            steps_arr.reshape((1, -1)),
            seq_len_arr.reshape((-1, 1))
        )
        _decoded_mask_arr = torch.mul(
            batch_mask_arr, 
            torch.unsqueeze(mask_arr, 0)
        )
        _decoded_mask_arr = torch.unsqueeze(_decoded_mask_arr, 0)
        _decoded_mask_arr = _decoded_mask_arr + 1e-08

        if decoded_mask_arr is None:
            decoded_mask_arr = _decoded_mask_arr
        else:
            decoded_mask_arr = torch.add(
                decoded_mask_arr, 
                _decoded_mask_arr
            )

        encoded_arr = self.encoder(
            encoded_observed_arr,
            encoded_mask_arr
        )
        self.feature_points_arr = encoded_arr
        decoded_arr = self.decoder(
            decoded_observed_arr, 
            encoded_arr, 
            decoded_mask_arr,
            encoded_mask_arr,
        )
        if self.output_nn is not None:
            decoded_arr = self.output_nn(decoded_arr)

        return decoded_arr

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = {}
        for k in self.state_dict().keys():
            params_dict.setdefault(k, self.state_dict()[k])

        return params_dict

    def regularize(self):
        '''
        Regularization.
        '''
        if len(self.__regularizatable_data_list) > 0:
            params_dict = self.extract_learned_dict()
            for regularizatable in self.__regularizatable_data_list:
                params_dict = regularizatable.regularize(params_dict)

            for k, params in params_dict.items():
                self.load_state_dict({k: params}, strict=False)

    def __rename_file(self, filename):
        filename_list = filename.split(".")
        _format = filename_list[-1]
        encoder_filename = filename.replace("." + _format, "_encoder." + _format)
        decoder_filename = filename.replace("." + _format, "_decoder." + _format)
        return encoder_filename, decoder_filename

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        encoder_filename, decoder_filename = self.__rename_file(filename)
        self.encoder.save_parameters(encoder_filename)
        self.decoder.save_parameters(decoder_filename)
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'loss': self.loss_arr,
            }, 
            filename
        )

    def load_parameters(self, filename, ctx=None, strict=True):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            Context-manager that changes the selected device.
            strict:         Whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict() function. Default: `True`.
        '''
        try:
            encoder_filename, decoder_filename = self.__rename_file(filename)
            self.encoder.load_parameters(encoder_filename, ctx=ctx, strict=strict)
            self.decoder.load_parameters(decoder_filename, ctx=ctx, strict=strict)

            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )
            self.epoch = checkpoint['epoch']
            self.__loss_list = checkpoint['loss'].tolist()
        except RuntimeError:
            self.__loaded_filename = filename
            self.__loaded_ctx = ctx

        if ctx is not None:
            self.to(ctx)
            self.encoder.to(ctx)
            self.decoder.to(ctx)
            self.__ctx = ctx

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

    __loss_list = []

    def get_loss_arr(self):
        ''' getter '''
        return np.array(self.__loss_list)
    
    def set_loss_arr(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    loss_arr = property(get_loss_arr, set_loss_arr)
