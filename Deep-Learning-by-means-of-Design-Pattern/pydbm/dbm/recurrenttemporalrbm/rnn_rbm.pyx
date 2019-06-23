# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.dbm.rtrbm_director import RTRBMDirector
from pydbm.dbm.builders.lstm_rt_rbm_simple_builder import LSTMRTRBMSimpleBuilder
from pydbm.approximation.rtrbmcd.lstm_rt_rbm_cd import LSTMRTRBMCD
from pydbm.optimization.opt_params import OptParams
from pydbm.dbm.recurrent_temporal_rbm import RecurrentTemporalRBM
from pydbm.params_initializer import ParamsInitializer


class RNNRBM(RecurrentTemporalRBM):
    '''
    The `Client` in Builder Pattern, to build RNN-RBM.

    The RTRBM can be understood as a sequence of conditional RBMs 
    whose parameters are the output of a deterministic RNN, 
    with the constraint that the hidden units must describe 
    the conditional distributions and convey temporal information. 
    This constraint can be lifted by combining a full RNN with distinct hidden units.
    
    RNN-RBM (Boulanger-Lewandowski, N., et al. 2012), which is the more 
    structural expansion of RTRBM, has also hidden units.

    References:
        - Boulanger-Lewandowski, N., Bengio, Y., & Vincent, P. (2012). Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. arXiv preprint arXiv:1206.6392.
        - Lyu, Q., Wu, Z., Zhu, J., & Meng, H. (2015, June). Modelling High-Dimensional Sequences with LSTM-RTRBM: Application to Polyphonic Music Generation. In IJCAI (pp. 4138-4139).
        - Lyu, Q., Wu, Z., & Zhu, J. (2015, October). Polyphonic music modelling with LSTM-RTRBM. In Proceedings of the 23rd ACM international conference on Multimedia (pp. 991-994). ACM.
        - Sutskever, I., Hinton, G. E., & Taylor, G. W. (2009). The recurrent temporal restricted boltzmann machine. In Advances in Neural Information Processing Systems (pp. 1601-1608).

    '''

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
            opt_params:                     is-a `OptParams`.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            pre_learned_path:               File path that stores pre-learned parameters.
            scale:                          Scale of parameters which will be `ParamsInitializer`.
            params_initializer:             is-a `ParamsInitializer`.
            params_dict:                    `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.

        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        if isinstance(opt_params, OptParams) is False:
            raise TypeError()

        rtrbm_director = RTRBMDirector(LSTMRTRBMSimpleBuilder(pre_learned_path))
        rtrbm_director.rtrbm_construct(
            visible_num,
            hidden_num,
            visible_activating_function,
            hidden_activating_function,
            rnn_activating_function,
            LSTMRTRBMCD(
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
