# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.rtrbm_director import RTRBMDirector
from pydbm.dbm.builders.rt_rbm_simple_builder import RTRBMSimpleBuilder
from pydbm.approximation.rt_rbm_cd import RTRBMCD
from pydbm.optimization.opt_params import OptParams
from pydbm.params_initializer import ParamsInitializer


class RecurrentTemporalRBM(object):
    '''
    The `Client` in Builder Pattern for building RTRBM.

    The RTRBM (Sutskever, I., et al. 2009) is a probabilistic 
    time-series model which can be viewed as a temporal stack of RBMs, 
    where each RBM has a contextual hidden state that is received 
    from the previous RBM and is used to modulate its hidden units bias.
    
    References:
        - Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. arXiv preprint arXiv:1206.6392.
        - Lyu, Q., Wu, Z., Zhu, J., & Meng, H. (2015, June). Modelling High-Dimensional Sequences with LSTM-RTRBM: Application to Polyphonic Music Generation. In IJCAI (pp. 4138-4139).
        - Lyu, Q., Wu, Z., & Zhu, J. (2015, October). Polyphonic music modelling with LSTM-RTRBM. In Proceedings of the 23rd ACM international conference on Multimedia (pp. 991-994). ACM.
        - Sutskever, I., Hinton, G. E., & Taylor, G. W. (2009). The recurrent temporal restricted boltzmann machine. In Advances in Neural Information Processing Systems (pp. 1601-1608).
    '''

    # Restricted Boltzmann Machine.
    __rbm = None
    
    def get_rbm(self):
        ''' getter '''
        return self.__rbm

    def set_rbm(self, value):
        ''' setter '''
        self.__rbm = value
    
    rbm = property(get_rbm, set_rbm)

    def __init__(
        self,
        visible_num,
        hidden_num,
        visible_activating_function,
        hidden_activating_function,
        rnn_activating_function,
        opt_params,
        learning_rate=1e-05,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        pre_learned_path=None,
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Init.

        Args:
            visible_num:                    The number of units in visible layer.
            hidden_num:                     The number of units in hidden layer.
            visible_activating_function:    The activation function in visible layer.
            hidden_activating_function:     The activation function in hidden layer.
            rnn_activating_function:        The activation function in RNN layer.
            opt_params:                     is-a `OptParams`.
            learning_rate:                  Learning rate.
            pre_learned_path:               File path that stores pre-learned parameters.
            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        if isinstance(opt_params, OptParams) is False:
            raise TypeError()

        rtrbm_director = RTRBMDirector(RTRBMSimpleBuilder(pre_learned_path))
        rtrbm_director.rtrbm_construct(
            visible_num,
            hidden_num,
            visible_activating_function,
            hidden_activating_function,
            rnn_activating_function,
            RTRBMCD(
                opt_params=opt_params
            ),
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            scale=scale,
            params_initializer=params_initializer,
            params_dict=params_dict
        )
        self.rbm = rtrbm_director.rbm

    def learn(self, observed_arr, training_count=1000, batch_size=200):
        '''
        Learning.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            training_count: The number of training.
            batch_size:     Batch size.
        '''
        # Learning.
        self.rbm.learn(
            # The `np.ndarray` of observed data points.
            observed_arr,
            # Training count.
            training_count=training_count, 
            # Batch size.
            batch_size=batch_size
        )

    def inference(self, test_arr, training_count=1, batch_size=None, r_batch_size=-1):
        '''
        Inferencing and recursive learning.
        
        Args:
            test_arr:           `np.ndarray` of test data points.
            training_count:     The number of training.
            batch_size:         Batch size.
            r_batch_size:       Batch size for recursive learning.

        Returns:
            `np.ndarray` of inferenced result.
        '''
        # Execute recursive learning.
        inferenced_arr = self.rbm.inference(
            test_arr,
            training_count=training_count, 
            r_batch_size=r_batch_size,
            batch_size=batch_size
        )
        return inferenced_arr

    def save_pre_learn_params(self, file_path):
        '''
        Save pre-learned parameters.
        
        Args:
            file_path:  Stored file path.
        '''
        self.rbm.graph.save_pre_learned_params(file_path)
