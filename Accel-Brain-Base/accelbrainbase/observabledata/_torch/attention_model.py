# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.iteratable_data import IteratableData
from accelbrainbase.regularizatable_data import RegularizatableData
from accelbrainbase.computable_loss import ComputableLoss
import numpy as np
from logging import getLogger
import torch
from torch import nn
from torch.optim.adam import Adam


class AttentionModel(nn.Module, ObservableData):
    '''
    Attention Model.

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

    __depth_dim = None

    def get_depth_dim(self):
        ''' getter '''
        return self.__depth_dim
    
    def set_depth_dim(self, value):
        ''' setter '''
        self.__depth_dim = value

    depth_dim = property(get_depth_dim, set_depth_dim)

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
        output_nn=None,
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
        if isinstance(depth_dim, int) is False:
            raise TypeError("The type of `depth_dim` must be `int`.")
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(AttentionModel, self).__init__()
        self.__computable_loss = computable_loss
        self.initializer_f = initializer_f
        self.optimizer_f = optimizer_f
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.output_nn = output_nn

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx
        self.ctx = ctx
        self.__learning_rate = learning_rate

        self.depth_dim = depth_dim

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__input_dim = None
        self.__not_init_flag = not_init_flag

        self.epoch = 0

    def initialize_params(self, input_dim):
        if self.__input_dim is not None:
            return

        self.__input_dim = input_dim

        self.query_dense_layer = nn.Linear(
            input_dim,
            self.depth_dim,
            bias=False,
        )
        if self.initializer_f is not None:
            self.query_dense_layer.weight = self.initializer_f(
                self.query_dense_layer.weight
            )
        else:
            self.query_dense_layer.weight = torch.nn.init.xavier_normal_(
                self.query_dense_layer.weight,
                gain=1.0
            )

        self.key_dense_layer = nn.Linear(
            input_dim,
            self.depth_dim,
            bias=False,
        )
        if self.initializer_f is not None:
            self.key_dense_layer.weight = self.initializer_f(
                self.key_dense_layer.weight
            )
        else:
            self.key_dense_layer.weight = torch.nn.init.xavier_normal_(
                self.key_dense_layer.weight,
                gain=1.0
            )

        self.value_dense_layer = nn.Linear(
            input_dim,
            self.depth_dim,
            bias=False,
        )
        if self.initializer_f is not None:
            self.value_dense_layer.weight = self.initializer_f(
                self.value_dense_layer.weight
            )
        else:
            self.value_dense_layer.weight = torch.nn.init.xavier_normal_(
                self.value_dense_layer.weight,
                gain=1.0
            )

        self.output_dense_layer = nn.Linear(
            self.depth_dim,
            self.depth_dim,
            bias=False,
        )
        if self.initializer_f is not None:
            self.output_dense_layer.weight = self.initializer_f(
                self.output_dense_layer.weight
            )
        else:
            self.output_dense_layer.weight = torch.nn.init.xavier_normal_(
                self.output_dense_layer.weight,
                gain=1.0
            )

        self.to(self.__ctx)

        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                if self.optimizer_f is not None:
                    self.optimizer = self.optimizer_f(
                        self.parameters()
                    )
                else:
                    self.optimizer = Adam(
                        self.parameters(), 
                        lr=self.__learning_rate,
                    )

    def learn(self, iteratable_data):
        '''
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

        Args:
            iteratable_data:     is-a `IteratableData`.
        '''
        if isinstance(iteratable_data, IteratableData) is False:
            raise TypeError("The type of `iteratable_data` must be `IteratableData`.")

        self.__loss_list = []
        learning_rate = self.__learning_rate

        pre_batch_observed_arr = None
        pre_test_batch_observed_arr = None
        try:
            epoch = self.epoch
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                # Self-Attention.
                self.__batch_size = batch_observed_arr.shape[0]
                self.__seq_len = batch_observed_arr.shape[1]
                if len(batch_observed_arr.shape) == 2:
                    batch_observed_arr = torch.unsqueeze(batch_observed_arr, axis=1)
                elif len(batch_observed_arr.shape) > 3:
                    batch_observed_arr = batch_observed_arr.reshape((
                        self.__batch_size,
                        self.__seq_len,
                        -1
                    ))

                self.initialize_params(
                    input_dim=batch_observed_arr.shape[2]
                )
                self.epoch = epoch

                if self.output_nn is not None:
                    if hasattr(self.output_nn, "optimizer") is False:
                        _ = self.inference(batch_observed_arr, batch_observed_arr)

                self.optimizer.zero_grad()
                if self.output_nn is not None:
                    self.output_nn.optimizer.zero_grad()

                pred_arr = self.inference(batch_observed_arr, batch_observed_arr)
                loss = self.compute_loss(
                    pred_arr,
                    batch_target_arr
                )
                loss.backward()
                if self.output_nn is not None:
                    self.output_nn.optimizer.step()
                self.optimizer.step()
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    with torch.inference_mode():
                        if len(test_batch_observed_arr.shape) == 2:
                            test_batch_observed_arr = torch.unsqueeze(test_batch_observed_arr, axis=1)

                        test_pred_arr = self.inference(test_batch_observed_arr, test_batch_observed_arr)

                        test_loss = self.compute_loss(
                            test_pred_arr,
                            test_batch_target_arr
                        )
                    _loss = loss.to('cpu').detach().numpy().copy()
                    _test_loss = test_loss.to('cpu').detach().numpy().copy()

                    self.__loss_list.append((_loss, _test_loss))
                    self.__logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(_loss) + " Test loss: " + str(_test_loss))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")
        self.epoch = epoch

    def inference(self, observed_arr, memory_arr):
        '''
        Inference samples drawn by `IteratableData.generate_inferenced_samples()`.

        Args:
            observed_arr:   Array like or sparse matrix as the observed data points.
            memory_arr:     Array like or sparse matrix as the observed data points.

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(observed_arr, memory_arr)

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

    def forward(self, x, m):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
            m:      `mxnet.ndarray` of observed data points. The shape is (batch_size, length of memory, depth).
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # Self-Attention.
        self.__batch_size = x.shape[0]
        self.__seq_len = x.shape[1]
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, axis=1)
        elif len(x.shape) > 3:
            x = x.reshape((
                self.__batch_size,
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
        logit_arr = torch.bmm(query_arr, key_arr.reshape((
            key_arr.shape[0],
            key_arr.shape[2],
            key_arr.shape[1]
        )))
        attention_weight_arr = self.softmax(logit_arr)
        attention_weight_arr = self.dropout(attention_weight_arr)
        attention_output_arr = torch.bmm(
            attention_weight_arr, 
            value_arr.reshape((
                value_arr.shape[0],
                value_arr.shape[2],
                value_arr.shape[1]
            ))
        )
        output_arr = self.output_dense_layer(attention_output_arr)

        if self.output_nn is not None:
            output_arr = self.output_nn(output_arr)

        return output_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_loss_arr(self):
        ''' getter for losses. '''
        return np.array(self.__loss_list)

    loss_arr = property(get_loss_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not. '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
