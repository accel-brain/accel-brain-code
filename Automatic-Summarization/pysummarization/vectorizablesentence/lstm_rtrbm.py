# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pysummarization.vectorizable_sentence import VectorizableSentence

# `Builder` in `Builder Patter`.
from pydbm.dbm.builders.lstm_rt_rbm_simple_builder import LSTMRTRBMSimpleBuilder
# LSTM and Contrastive Divergence for function approximation.
from pydbm.approximation.rtrbmcd.lstm_rt_rbm_cd import LSTMRTRBMCD
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# Stochastic Gradient Descent(SGD) as optimizer.
from pydbm.optimization.optparams.sgd import SGD


class LSTMRTRBM(VectorizableSentence):
    '''
    Vectorize sentences by LSTM-RTRBM.
    
    LSTM-RTRBM model integrates the ability of LSTM in memorizing 
    and retrieving useful history information, together with the 
    advantage of RBM in high dimensional data 
    modelling(Lyu, Q., Wu, Z., Zhu, J., & Meng, H. 2015, June). 
    Like RTRBM, LSTM-RTRBM also has the recurrent hidden units.
    
    References:
        - Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. arXiv preprint arXiv:1206.6392.
        - Lyu, Q., Wu, Z., Zhu, J., & Meng, H. (2015, June). Modelling High-Dimensional Sequences with LSTM-RTRBM: Application to Polyphonic Music Generation. In IJCAI (pp. 4138-4139).
        - Lyu, Q., Wu, Z., & Zhu, J. (2015, October). Polyphonic music modelling with LSTM-RTRBM. In Proceedings of the 23rd ACM international conference on Multimedia (pp. 991-994). ACM.
        - Sutskever, I., Hinton, G. E., & Taylor, G. W. (2009). The recurrent temporal restricted boltzmann machine. In Advances in Neural Information Processing Systems (pp. 1601-1608).

    '''

    def vectorize(self, sentence_list):
        '''
        Args:
            sentence_list:   The list of tokenized sentences.
                             [[`token`, `token`, `token`, ...],
                             [`token`, `token`, `token`, ...],
                             [`token`, `token`, `token`, ...]]
        
        Returns:
            `np.ndarray` of tokens.
            [vector of token, vector of token, vector of token]
        '''
        test_observed_arr = self.__setup_dataset(sentence_list, self.__token_master_list, self.__seq_len)

        inferenced_arr = self.__rbm.inference(
            test_observed_arr,
            training_count=1, 
            r_batch_size=-1
        )

        return inferenced_arr

    def learn(
        self,
        sentence_list,
        token_master_list,
        hidden_neuron_count=1000,
        training_count=1,
        batch_size=100,
        learning_rate=1e-03,
        seq_len=5
    ):
        '''
        Init.
        
        Args:
            sentence_list:                  The `list` of sentences.
            token_master_list:              Unique `list` of tokens.
            hidden_neuron_count:            The number of units in hidden layer.
            training_count:                 The number of training.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            seq_len:                        The length of one sequence.
        '''
        observed_arr = self.__setup_dataset(sentence_list, token_master_list, seq_len)

        visible_num = observed_arr.shape[-1]

        # `Builder` in `Builder Pattern` for LSTM-RTRBM.
        rnnrbm_builder = LSTMRTRBMSimpleBuilder()
        # Learning rate.
        rnnrbm_builder.learning_rate = learning_rate
        # Set units in visible layer.
        rnnrbm_builder.visible_neuron_part(LogisticFunction(), visible_num)
        # Set units in hidden layer.
        rnnrbm_builder.hidden_neuron_part(LogisticFunction(), hidden_neuron_count) 
        # Set units in RNN layer.
        rnnrbm_builder.rnn_neuron_part(TanhFunction())
        # Set graph and approximation function, delegating `SGD` which is-a `OptParams`.
        rnnrbm_builder.graph_part(LSTMRTRBMCD(opt_params=SGD()))
        # Building.
        rbm = rnnrbm_builder.get_result()
        
        # Learning.
        rbm.learn(
            # The `np.ndarray` of observed data points.
            observed_arr,
            # Training count.
            training_count=training_count, 
            # Batch size.
            batch_size=batch_size
        )
        
        self.__rbm = rbm
        self.__token_master_list = token_master_list
        self.__seq_len = seq_len

    def __setup_dataset(self, sentence_list, token_master_list, seq_len):
        sentence_len_list = [0] * len(sentence_list)
        for i in range(len(sentence_list)):
            sentence_len_list[i] = len(sentence_list[i])

        observed_list = [None] * len(sentence_list)
        for i in range(len(sentence_list)):
            arr_list = [None] * seq_len
            for j in range(seq_len):
                arr = np.zeros(len(token_master_list))
                try:
                    token = sentence_list[i][j]
                    arr[token_master_list.index(token)] = 1
                except IndexError:
                    pass
                finally:
                    arr = arr.astype(np.float64)
                    arr_list[j] = arr
            observed_list[i] = arr_list
        observed_arr = np.array(observed_list)
        return observed_arr
