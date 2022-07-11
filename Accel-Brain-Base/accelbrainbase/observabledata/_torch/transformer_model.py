# -*- coding: utf-8 -*-
from accelbrainbase.observable_data import ObservableData
from accelbrainbase.computable_loss import ComputableLoss

from abc import abstractmethod, abstractproperty
import numpy as np
import torch


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

    __optimizer = None

    def get_optimizer(self):
        return self.__optimizer
    
    def set_optimizer(self, value):
        self.__optimizer = value

    optimizer = property(get_optimizer, set_optimizer)

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
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    optimizer_setup_flag = True
                else:
                    optimizer_setup_flag = False

                # Self-Attention.
                if len(batch_observed_arr.shape) == 2:
                    batch_observed_arr = torch.unsqueeze(batch_observed_arr, axis=1)

                pred_arr = self.inference(batch_observed_arr, batch_observed_arr)
                loss = self.compute_loss(
                    pred_arr,
                    batch_target_arr
                )
                if optimizer_setup_flag is False:
                    # After initilization, restart.
                    self.optimizer.zero_grad()
                    pred_arr = self.inference(batch_observed_arr, batch_observed_arr)
                    loss = self.compute_loss(
                        pred_arr,
                        test_batch_target_arr
                    )

                loss.backward()
                self.optimizer.step()
                self.regularize()

                if (iter_n+1) % int(iteratable_data.iter_n / iteratable_data.epochs) == 0:
                    with torch.inference_mode():
                        if len(test_batch_observed_arr.shape) == 2:
                            test_batch_observed_arr = torch.unsqueeze(test_batch_observed_arr, axis=1)

                        test_pred_arr = self.inference(test_batch_observed_arr, test_batch_observed_arr)

                        test_loss = self.compute_loss(
                            test_pred_arr,
                            test_batch_observed_arr
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
        if len(self.regularizatable_data_list) > 0:
            params_dict = self.extract_learned_dict()
            for regularizatable in self.regularizatable_data_list:
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

        arr = torch.from_numpy(np.arange(depth_dim))
        arr = arr.to(observed_arr.device)
        depth_arr = torch.tile(
            torch.unsqueeze(
                (
                    arr / 2
                ).to(torch.int32) * 2,
                0
            ), 
            (seq_len, 1)
        )

        depth_arr = depth_arr / depth_dim
        depth_arr = torch.pow(10000.0, depth_arr).to(torch.float32)

        arr = torch.from_numpy(np.arange(depth_dim))
        arr = arr.to(observed_arr.device)
        phase_arr = torch.tile(
            torch.unsqueeze(
                (
                    arr % 2
                ) * np.pi / 2,
                0
            ), 
            (seq_len, 1)
        )
        arr = torch.from_numpy(np.arange(seq_len))
        arr = arr.to(observed_arr.device)
        positional_arr = torch.tile(
            torch.unsqueeze(
                arr, 
                1
            ), 
            (1, depth_dim)
        )

        sin_arr = torch.sin(positional_arr / depth_arr + phase_arr)

        positional_encoded_arr = torch.tile(
            torch.unsqueeze(sin_arr, 0), 
            (batch_size, 1, 1)
        )

        result_arr = observed_arr + (positional_encoded_arr * self.embedding_weignt)
        return result_arr

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
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
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.epoch = checkpoint['epoch']
        self.__loss_list = checkpoint['loss'].tolist()
        if ctx is not None:
            self.to(ctx)
            self.__ctx = ctx

    def get_loss_arr(self):
        ''' getter for losses. '''
        return self.__loss_arr

    def set_loss_arr(self, value):
        self.__loss_arr = value

    loss_arr = property(get_loss_arr, set_loss_arr)

    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not. '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not. '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
