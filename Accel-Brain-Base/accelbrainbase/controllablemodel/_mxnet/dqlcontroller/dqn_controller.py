# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from accelbrainbase.controllablemodel._mxnet.dql_controller import DQLController


class DQNController(DQLController):
    '''
    Abstract base class to implement the Deep Q-Network(DQN).

    The structure of Q-Learning is based on the Epsilon Greedy Q-Leanring algorithm,
    which is a typical off-policy algorithm.  In this paradigm, stochastic searching 
    and deterministic searching can coexist by hyperparameter `epsilon_greedy_rate` 
    that is probability that agent searches greedy. Greedy searching is deterministic 
    in the sensethat policy of agent follows the selection that maximizes the Q-Value.

    References:
        - https://code.accel-brain.com/Reinforcement-Learning/README.html#deep-q-network
        - Egorov, M. (2016). Multi-agent deep reinforcement learning.(URL: https://pdfs.semanticscholar.org/dd98/9d94613f439c05725bad958929357e365084.pdf)
        - Gupta, J. K., Egorov, M., & Kochenderfer, M. (2017, May). Cooperative multi-agent control using deep reinforcement learning. In International Conference on Autonomous Agents and Multiagent Systems (pp. 66-83). Springer, Cham.
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
    '''

    # epsilo-greedy rate.
    __epsilon_greedy_rate = 0.75

    def select_action(
        self, 
        possible_action_arr, 
        possible_predicted_q_arr, 
        possible_reward_value_arr,
        possible_next_q_arr,
        possible_meta_data_arr=None
    ):
        '''
        Select action by Q(state, action).

        Args:
            possible_action_arr:                Tensor of actions.
            possible_predicted_q_arr:           Tensor of Q-Values.
            possible_reward_value_arr:          Tensor of reward values.
            possible_next_q_arr:                Tensor of Q-Values in next time.
            possible_meta_data_arr:             `mxnet.ndarray.NDArray` or `np.array` of meta data of the actions.

        Retruns:
            Tuple(`np.ndarray` of action., Q-Value)
        '''
        key_arr = self.select_action_key(possible_action_arr, possible_predicted_q_arr)
        meta_data_arr = None
        if possible_meta_data_arr is not None:
            for i in range(possible_meta_data_arr.shape[0]):
                _meta_data_arr = possible_meta_data_arr[i, key_arr[i]]
                if i == 0:
                    if isinstance(_meta_data_arr, nd.NDArray) is True:
                        meta_data_arr = nd.expand_dims(_meta_data_arr, axis=0)
                    else:
                        meta_data_arr = np.expand_dims(_meta_data_arr, axis=0)
                else:
                    if isinstance(_meta_data_arr, nd.NDArray) is True:
                        meta_data_arr = nd.concat(
                            meta_data_arr,
                            nd.expand_dims(_meta_data_arr, axis=0),
                            dim=0
                        )
                    else:
                        meta_data_arr = np.concatenate(
                            [
                                meta_data_arr,
                                np.expand_dims(_meta_data_arr, axis=0),
                            ],
                            axis=0
                        )

        action_arr = None
        predicted_q_arr = None
        reward_value_arr = None
        next_q_arr = None

        for i in range(possible_action_arr.shape[0]):
            _action_arr = possible_action_arr[i, key_arr[i]]
            _predicted_q_arr = possible_predicted_q_arr[i, key_arr[i]]
            _reward_value_arr = possible_reward_value_arr[i, key_arr[i]]
            _next_q_arr = possible_next_q_arr[i, key_arr[i]]
            if i == 0:
                action_arr = nd.expand_dims(_action_arr, axis=0)
                predicted_q_arr = nd.expand_dims(_predicted_q_arr, axis=0)
                reward_value_arr = nd.expand_dims(_reward_value_arr, axis=0)
                next_q_arr = nd.expand_dims(_next_q_arr, axis=0)
            else:
                action_arr = nd.concat(
                    action_arr,
                    nd.expand_dims(_action_arr, axis=0),
                    dim=0
                )
                predicted_q_arr = nd.concat(
                    predicted_q_arr,
                    nd.expand_dims(_predicted_q_arr, axis=0),
                    dim=0
                )
                reward_value_arr = nd.concat(
                    reward_value_arr,
                    nd.expand_dims(_reward_value_arr, axis=0),
                    dim=0
                )
                next_q_arr = nd.concat(
                    next_q_arr,
                    nd.expand_dims(_next_q_arr, axis=0),
                    dim=0
                )

        return (
            action_arr, 
            predicted_q_arr, 
            reward_value_arr,
            next_q_arr,
            meta_data_arr
        )

    def select_action_key(self, possible_action_arr, possible_predicted_q_arr):
        '''
        Select action by Q(state, action).

        Args:
            possible_action_arr:        `np.ndarray` of actions.
            possible_predicted_q_arr:             `np.ndarray` of Q-Values.

        Retruns:
            `np.ndarray` of keys.
        '''
        epsilon_greedy_flag = bool(np.random.binomial(n=1, p=self.epsilon_greedy_rate))
        if epsilon_greedy_flag is False:
            key_arr = np.random.randint(
                size=(possible_action_arr.shape[0], ),
                low=0, 
                high=possible_action_arr.shape[1]
            )
        else:
            key_arr = possible_predicted_q_arr[:, :, 0].argmax(axis=1).asnumpy().astype(int)

        return key_arr

    def get_epsilon_greedy_rate(self):
        ''' getter for epsilo-greedy rate '''
        if isinstance(self.__epsilon_greedy_rate, float) is True:
            return self.__epsilon_greedy_rate
        else:
            raise TypeError("The type of __epsilon_greedy_rate must be float.")

    def set_epsilon_greedy_rate(self, value):
        ''' setter for epsilo-greedy rate '''
        if isinstance(value, float) is True:
            self.__epsilon_greedy_rate = value
        else:
            raise TypeError("The type of __epsilon_greedy_rate must be float.")

    epsilon_greedy_rate = property(get_epsilon_greedy_rate, set_epsilon_greedy_rate)
