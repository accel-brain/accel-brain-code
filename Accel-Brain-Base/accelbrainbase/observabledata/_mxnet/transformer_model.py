# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.computable_loss import ComputableLoss

from abc import abstractmethod
import mxnet.ndarray as nd
import mxnet as mx
import numpy as np


class TransformerModel(ObservableData):
    '''
    The abstract class of Transformer.

    References:
        - Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
        - Floridi, L., & Chiriatti, M. (2020). GPT-3: Its nature, scope, limits, and consequences. Minds and Machines, 30(4), 681-694.
        - Miller, A., Fisch, A., Dodge, J., Karimi, A. H., Bordes, A., & Weston, J. (2016). Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126.
        - Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018) Improving Language Understanding by Generative Pre-Training. OpenAI (URL: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
        - Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.

    '''

    __embedding_flag = True

    def get_embedding_flag(self):
        ''' getter '''
        return self.__embedding_flag
    
    def set_embedding_flag(self, value):
        ''' setter '''
        self.__embedding_flag = value

    embedding_flag = property(get_embedding_flag, set_embedding_flag)

    __embedding_weignt = 1.0

    def get_embedding_weignt(self):
        ''' getter '''
        return self.__embedding_weignt
    
    def set_embedding_weignt(self, value):
        ''' setter '''
        self.__embedding_weignt = value
    
    embedding_weignt = property(get_embedding_weignt, set_embedding_weignt)

    def learn(self, iteratable_data):
        '''
        Learn samples drawn by `IteratableData.generate_learned_samples()`.

        Args:
            iteratable_data:     is-a `IteratableData`.
        '''
        if isinstance(iteratable_data, IteratableData) is False:
            raise TypeError("The type of `iteratable_data` must be `IteratableData`.")

        self.__loss_list = []
        learning_rate = self.learning_rate

        pre_batch_observed_arr = None
        pre_test_batch_observed_arr = None
        try:
            epoch = 0
            iter_n = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in iteratable_data.generate_learned_samples():
                self.epoch = epoch
                if ((epoch + 1) % self.attenuate_epoch == 0):
                    learning_rate = learning_rate * self.learning_attenuate_rate
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
                    self.logger.debug("Epochs: " + str(epoch + 1) + " Train loss: " + str(loss.asnumpy().mean()) + " Test loss: " + str(test_loss.asnumpy().mean()))
                    epoch += 1
                iter_n += 1

        except KeyboardInterrupt:
            self.logger.debug("Interrupt.")

        self.logger.debug("end. ")

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

    def embedding(self, observed_arr):
        '''
        Embedding. In default, this method does the positional encoding.

        Args:
            observed_arr:       `mxnet.ndarray` of observed data points.

        Returns:
            `mxnet.ndarray` of embedded data points.
        '''
        batch_size, seq_len, depth_dim = observed_arr.shape
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.depth_dim = depth_dim

        if self.embedding_flag is False:
            return observed_arr

        depth_arr = nd.tile(nd.expand_dims((nd.arange(depth_dim) / 2).astype(int) * 2, 0), [seq_len, 1])

        depth_arr = depth_arr / depth_dim
        depth_arr = nd.power(10000.0, depth_arr).astype(np.float32)

        phase_arr = nd.tile(nd.expand_dims((nd.arange(depth_dim) % 2) * np.pi / 2, 0), [seq_len, 1])
        positional_arr = nd.tile(nd.expand_dims(nd.arange(seq_len), 1), [1, depth_dim])

        sin_arr = nd.sin(positional_arr / depth_arr + phase_arr)

        positional_encoded_arr = nd.tile(nd.expand_dims(sin_arr, 0), [batch_size, 1, 1])

        positional_encoded_arr = positional_encoded_arr.as_in_context(observed_arr.context)

        result_arr = observed_arr + (positional_encoded_arr * self.embedding_weignt)
        return result_arr

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not. '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)

    # is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
    __initializer = None

    def get_initializer(self):
        ''' getter for `mxnet.initializer`. '''
        return self.__initializer
    
    def set_initializer(self, value):
        ''' setter for `mxnet.initializer`. '''
        self.__initializer = value
    
    initializer = property(get_initializer, set_initializer)
