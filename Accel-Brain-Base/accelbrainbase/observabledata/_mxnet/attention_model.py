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


class AttentionModel(HybridBlock, ObservableData):
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
        dropout_rate=0.5,
        initializer=None,
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        optimizer_name="SGD",
        ctx=mx.gpu(),
        hybridize_flag=True,
        regularizatable_data_list=[],
        not_init_flag=False,
        **kwargs
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
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, gluon.loss.Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(AttentionModel, self).__init__()
        self.__computable_loss = computable_loss

        if initializer is None:
            self.initializer = mx.initializer.Normal(1.0)
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")

            self.initializer = initializer

        with self.name_scope():
            self.query_dense_layer = gluon.nn.Dense(
                depth_dim,
                use_bias=False,
                flatten=False,
            )
            self.key_dense_layer = gluon.nn.Dense(
                depth_dim,
                use_bias=False,
                flatten=False,
            )
            self.value_dense_layer = gluon.nn.Dense(
                depth_dim,
                use_bias=False,
                flatten=False,
            )
            self.dropout = gluon.nn.Dropout(rate=dropout_rate)

            self.output_dense_layer = gluon.nn.Dense(
                depth_dim,
                use_bias=False,
                flatten=False,
            )
            self.register_child(self.query_dense_layer)
            self.register_child(self.key_dense_layer)
            self.register_child(self.value_dense_layer)
            self.register_child(self.output_dense_layer)
            self.register_child(self.dropout)

        if self.init_deferred_flag is False:
            if not_init_flag is False:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(self.collect_params(), optimizer_name, {"learning_rate": learning_rate})
                if hybridize_flag is True:
                    self.hybridize()

        for v in regularizatable_data_list:
            if isinstance(v, RegularizatableData) is False:
                raise TypeError("The type of values of `regularizatable_data_list` must be `RegularizatableData`.")
        self.__regularizatable_data_list = regularizatable_data_list

        self.__ctx = ctx
        self.ctx = ctx
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__optimizer_name = optimizer_name

        self.depth_dim = depth_dim

        logger = getLogger("accelbrainbase")
        self.__logger = logger

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
            epoch = 0
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.trainer.set_learning_rate(learning_rate)

                with autograd.record():
                    # Self-Attention.
                    if batch_observed_arr.ndim == 2:
                        batch_observed_arr = nd.expand_dims(batch_observed_arr, axis=1)

                    pred_arr = self.inference(batch_observed_arr, batch_observed_arr)
                    loss = self.compute_loss(
                        pred_arr,
                        batch_observed_arr
                    )
                loss.backward()
                self.trainer.step(batch_observed_arr.shape[0])
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    if test_batch_observed_arr.ndim == 2:
                        test_batch_observed_arr = nd.expand_dims(test_batch_observed_arr, axis=1)

                    test_pred_arr = self.inference(test_batch_observed_arr, test_batch_observed_arr)

                    test_loss = self.compute_loss(
                        test_pred_arr,
                        test_batch_observed_arr
                    )
                    self.__loss_list.append((loss.asnumpy().mean(), test_loss.asnumpy().mean()))
                    self.__logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(loss.asnumpy().mean()) + " Test loss: " + str(test_loss.asnumpy().mean()))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        self.__logger.debug("end. ")

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
        params_dict = self.extract_learned_dict()
        for regularizatable in self.__regularizatable_data_list:
            params_dict = regularizatable.regularize(params_dict)

        for k, params in self.collect_params().items():
            params.set_data(params_dict[k])

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

    def hybrid_forward(self, F, x, m):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
            m:      `mxnet.ndarray` of observed data points. The shape is (batch_size, length of memory, depth).
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x, m)

    def forward_propagation(self, F, x, m):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of query. The shape is (batch_size, length of query, depth).
            m:      `mxnet.ndarray` of memory. The shape is (batch_size, length of memory, depth).
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        query_arr = self.query_dense_layer(x)
        key_arr = self.key_dense_layer(m)
        value_arr = self.value_dense_layer(m)
        logit_arr = F.batch_dot(query_arr, key_arr, transpose_b=True)
        attention_weight_arr = F.softmax(logit_arr)
        attention_weight_arr = self.dropout(attention_weight_arr)
        attention_output_arr = F.batch_dot(attention_weight_arr, value_arr)
        output_arr = self.output_dense_layer(attention_output_arr)
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

    def get_units_list(self):
        ''' getter for `list` of units in each layer. '''
        return self.__units_list
    
    units_list = property(get_units_list, set_readonly)

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`. '''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`.'''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
