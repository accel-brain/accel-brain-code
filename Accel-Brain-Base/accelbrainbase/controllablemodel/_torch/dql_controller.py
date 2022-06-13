# -*- coding: utf-8 -*-
from accelbrainbase.controllable_model import ControllableModel
from accelbrainbase.observabledata._torch.function_approximator import FunctionApproximator
from accelbrainbase.samplabledata.policy_sampler import PolicySampler
from accelbrainbase.computable_loss import ComputableLoss

import numpy as np
from logging import getLogger
from abc import abstractmethod
import torch
from torch import nn


class DQLController(nn.Module, ControllableModel):
    '''
    Abstract base class to implement the Deep Q-Learning.

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

    def __init__(
        self,
        function_approximator,
        policy_sampler,
        computable_loss,
        ctx="cpu",
        recursive_learning_flag=False,
    ):
        '''
        Init.

        Args:
            function_approximator:          is-a `FunctionApproximator`.
            policy_sampler:                 is-a `PolicySampler`.
            computable_loss:                is-a `ComputableLoss` or `mxnet.gluon.loss`.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.

        '''
        if isinstance(function_approximator, FunctionApproximator) is False:
            raise TypeError("The type of `function_approximator` must be `FunctionApproximator`.")
        if isinstance(policy_sampler, PolicySampler) is False:
            raise TypeError("The type of `policy_sampler` must be `PolicySampler`.")
        if isinstance(computable_loss, ComputableLoss) is False and isinstance(computable_loss, nn.modules.loss._Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")

        super(DQLController, self).__init__()

        self.__function_approximator = function_approximator
        self.function_approximator = self.__function_approximator
        self.__policy_sampler = policy_sampler
        self.__computable_loss = computable_loss

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__recursive_learning_flag = recursive_learning_flag
        self.__q_logs_arr = np.array([])

        self.t = 0
        self.epoch = 0

    def learn(
        self, 
        iter_n=100,
    ):
        '''
        Learning.

        Args:
            iter_n:     `int` of the number of training iterations.
        '''
        q_logs_list = []
        r_logs_list = []
        posterior_logs_list = []

        state_arr = None
        state_meta_data_arr = None

        try:
            for n in range(iter_n):
                if (n + 1) % 100 == 0 or n < 100:
                    self.__logger.debug("-" * 100)
                    self.__logger.debug("Iter: " + str(n + 1))

                # Draw samples of next possible actions from any distribution.
                # (batch, possible_n, dim1, dim2, ...)
                possible_action_arr, action_meta_data_arr = self.policy_sampler.draw()

                possible_reward_value_arr = None
                next_q_arr = None
                possible_predicted_q_arr = None

                for possible_i in range(possible_action_arr.shape[1]):
                    if action_meta_data_arr is not None:
                        meta_data_arr = action_meta_data_arr[:, possible_i]
                    else:
                        meta_data_arr = None

                    # Inference Q-Values.
                    _predicted_q_arr = self.function_approximator(
                        possible_action_arr[:, possible_i]
                    )
                    if possible_predicted_q_arr is None:
                        possible_predicted_q_arr = torch.unsqueeze(_predicted_q_arr, dim=1)
                    else:
                        possible_predicted_q_arr = torch.cat(
                            (
                                possible_predicted_q_arr,
                                torch.unsqueeze(_predicted_q_arr, dim=1),
                            ),
                            dim=1
                        )

                    # Observe reward values.
                    _reward_value_arr = self.policy_sampler.observe_reward_value(
                        state_arr, 
                        possible_action_arr[:, possible_i],
                        meta_data_arr=meta_data_arr,
                    )
                    if possible_reward_value_arr is None:
                        possible_reward_value_arr = torch.unsqueeze(_reward_value_arr, axis=1)
                    else:
                        possible_reward_value_arr = torch.cat(
                            (
                                possible_reward_value_arr,
                                torch.unsqueeze(_reward_value_arr, dim=1),
                            ),
                            dim=1
                        )

                    # Inference the Max-Q-Value in next action time.
                    self.policy_sampler.observe_state(
                        state_arr=possible_action_arr[:, possible_i],
                        meta_data_arr=meta_data_arr
                    )
                    next_possible_action_arr, _ = self.policy_sampler.draw()
                    next_next_q_arr = None

                    for possible_j in range(next_possible_action_arr.shape[1]):
                        with torch.inference_mode():
                            _next_next_q_arr = self.function_approximator(
                                next_possible_action_arr[:, possible_j]
                            )
                        if next_next_q_arr is None:
                            next_next_q_arr = torch.unsqueeze(
                                _next_next_q_arr,
                                dim=1
                            )
                        else:
                            next_next_q_arr = torch.cat(
                                (
                                    next_next_q_arr,
                                    torch.unsqueeze(
                                        _next_next_q_arr, 
                                        dim=1
                                    ),
                                ),
                                dim=1
                            )

                    next_max_q_arr = next_next_q_arr.max(dim=1, keepdim=True).values

                    if next_q_arr is None:
                        next_q_arr = torch.unsqueeze(
                            next_max_q_arr,
                            dim=1,
                        )
                    else:
                        next_q_arr = torch.cat(
                            (
                                next_q_arr,
                                torch.unsqueeze(
                                    next_max_q_arr,
                                    dim=1,
                                ),
                            ),
                            dim=1
                        )

                # Select action.
                selected_tuple = self.select_action(
                    possible_action_arr, 
                    possible_predicted_q_arr,
                    possible_reward_value_arr,
                    next_q_arr,
                    possible_meta_data_arr=action_meta_data_arr
                )
                action_arr, predicted_q_arr, reward_value_arr, next_q_arr, action_meta_data_arr = selected_tuple

                self.function_approximator.model.optimizer.zero_grad()

                predicted_q_arr = self.function_approximator(
                    action_arr
                )
                # Update real Q-Values.
                real_q_arr = self.update_q(
                    reward_value_arr,
                    next_q_arr
                )

                if self.__q_logs_arr.shape[0] > 0:
                    self.__q_logs_arr = np.r_[
                        self.__q_logs_arr,
                        np.array([
                            predicted_q_arr.to('cpu').detach().numpy().copy().mean(), 
                            real_q_arr.to('cpu').detach().numpy().copy().mean(),
                        ]).reshape(1, 2)
                    ]
                else:
                    self.__q_logs_arr = np.array([
                        predicted_q_arr.to('cpu').detach().numpy().copy().mean(), 
                        real_q_arr.to('cpu').detach().numpy().copy().mean()
                    ]).reshape(1, 2)

                # Learn Q-Values.
                loss = self.computable_loss(
                    predicted_q_arr, 
                    real_q_arr
                )
                if torch.isnan(loss).to('cpu').detach().numpy().copy().astype(int).sum() == 0:
                    loss.backward()
                    self.function_approximator.model.optimizer.step()
                    self.function_approximator.model.regularize()

                    if (n + 1) % 100 == 0 or n < 100:
                        self.__logger.debug("Reward value(mean): " + str(reward_value_arr.mean().to('cpu').detach().numpy().copy()))
                        self.__logger.debug("Predicted Q-value(mean): " + str(predicted_q_arr.mean().to('cpu').detach().numpy().copy()))
                        self.__logger.debug("Real Q-value(mean): " + str(real_q_arr.mean().to('cpu').detach().numpy().copy()))
                        self.__logger.debug("Loss of Q-Value(mean): " + str(loss.mean().to('cpu').detach().numpy().copy()))

                    if self.__recursive_learning_flag is True:
                        # Update State.
                        state_arr, state_meta_data_arr = self.policy_sampler.update_state(
                            action_arr, 
                            meta_data_arr=action_meta_data_arr
                        )
                        self.policy_sampler.observe_state(
                            state_arr=state_arr,
                            meta_data_arr=state_meta_data_arr
                        )
                else:
                    self.__logger.debug("The parameter update was skipped because the vanishing gradient problem or gradient explosion occurred.")

                # Epsode.
                self.t += 1

                if self.__recursive_learning_flag is True:
                    # Check.
                    end_flag = self.policy_sampler.check_the_end_flag(
                        state_arr, 
                        meta_data_arr=meta_data_arr
                    )
                    if end_flag is True:
                        break

        except KeyboardInterrupt:
            print("Keyboard Interrupt.")

    def inference(self, iter_n=100):
        '''
        Inference.

        Args:
            iter_n:     `int` of the number of training iterations.

        Returns:
            `list` of logs of states.
        '''
        state_arr_list = []
        q_value_arr_list = []
        state_arr = None
        try:
            for n in range(iter_n):
                # Draw samples of next possible actions from any distribution.
                # (batch, possible_n, dim1, dim2, ...)
                possible_action_arr, action_meta_data_arr = self.policy_sampler.draw()

                possible_reward_value_arr = None
                next_q_arr = None
                possible_predicted_q_arr = None

                for possible_i in range(possible_action_arr.shape[1]):
                    if action_meta_data_arr is not None:
                        meta_data_arr = action_meta_data_arr[:, possible_i]
                    else:
                        meta_data_arr = None

                    # Inference Q-Values.
                    _predicted_q_arr = self.function_approximator(
                        possible_action_arr[:, possible_i]
                    )
                    if possible_predicted_q_arr is None:
                        possible_predicted_q_arr = torch.unsqueeze(_predicted_q_arr, dim=1)
                    else:
                        possible_predicted_q_arr = torch.cat(
                            (
                                possible_predicted_q_arr,
                                torch.unsqueeze(_predicted_q_arr, dim=1),
                            ),
                            dim=1
                        )

                    # Observe reward values.
                    _reward_value_arr = self.policy_sampler.observe_reward_value(
                        state_arr, 
                        possible_action_arr[:, possible_i],
                        meta_data_arr=meta_data_arr,
                    )
                    if possible_reward_value_arr is None:
                        possible_reward_value_arr = torch.unsqueeze(_reward_value_arr, dim=1)
                    else:
                        possible_reward_value_arr = torch.cat(
                            (
                                possible_reward_value_arr,
                                torch.unsqueeze(_reward_value_arr, dim=1),
                            ),
                            dim=1
                        )

                    # Inference the Max-Q-Value in next action time.
                    self.policy_sampler.observe_state(
                        state_arr=possible_action_arr[:, possible_i],
                        meta_data_arr=meta_data_arr
                    )
                    next_possible_action_arr, _ = self.policy_sampler.draw()
                    next_next_q_arr = None

                    for possible_j in range(next_possible_action_arr.shape[1]):
                        _next_next_q_arr = self.function_approximator(
                            next_possible_action_arr[:, possible_j]
                        )
                        if next_next_q_arr is None:
                            next_next_q_arr = torch.unsqueeze(
                                _next_next_q_arr,
                                dim=1
                            )
                        else:
                            next_next_q_arr = torch.cat(
                                (
                                    next_next_q_arr,
                                    torch.unsqueeze(
                                        _next_next_q_arr, 
                                        dim=1
                                    ),
                                ),
                                dim=1
                            )

                    next_max_q_arr = next_next_q_arr.max(dim=1, keepdim=True).values

                    if next_q_arr is None:
                        next_q_arr = torch.unsqueeze(
                            next_max_q_arr,
                            dim=1
                        )
                    else:
                        next_q_arr = torch.cat(
                            (
                                next_q_arr,
                                torch.unsqueeze(
                                    next_max_q_arr,
                                    dim=1
                                ),
                            ),
                            dim=1
                        )

                # Select action.
                selected_tuple = self.select_action(
                    possible_action_arr, 
                    possible_predicted_q_arr,
                    possible_reward_value_arr,
                    next_q_arr,
                    possible_meta_data_arr=action_meta_data_arr
                )
                action_arr, predicted_q_arr, reward_value_arr, next_q_arr, action_meta_data_arr = selected_tuple

                # Update State.
                state_arr, state_meta_data_arr = self.policy_sampler.update_state(
                    action_arr, 
                    meta_data_arr=action_meta_data_arr
                )
                self.policy_sampler.observe_state(
                    state_arr=state_arr,
                    meta_data_arr=state_meta_data_arr
                )
                state_arr_list.append(state_arr)
                q_value_arr_list.append(predicted_q_arr)

                # Epsode.
                self.t += 1

                # Check.
                end_flag = self.policy_sampler.check_the_end_flag(
                    state_arr, 
                    meta_data_arr=meta_data_arr
                )
                if end_flag is True:
                    break

        except KeyboardInterrupt:
            print("Keyboard Interrupt.")

        return state_arr_list, q_value_arr_list

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = {}
        for k in self.function_approximator.model.state_dict().keys():
            params_dict.setdefault(k, self.function_approximator.model.state_dict()[k])

        return params_dict

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        self.function_approximator.model.save_parameters(filename)

    def load_parameters(self, filename, ctx=None, strict=True):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            `mx.cpu()` or `mx.gpu()`.
            allow_missing:  `bool` of whether to silently skip loading parameters not represents in the file.
            ignore_extra:   `bool` of whether to silently ignre parameters from the file that are not present in this `Block`.
        '''
        self.function_approximator.model.load_parameters(filename, ctx=ctx, strict=strict)

    @abstractmethod
    def select_action(
        self, 
        possible_action_arr, 
        possible_predicted_q_arr, 
        possible_reward_value_arr,
        possible_meta_data_arr=None
    ):
        '''
        Select action by Q(state, action).

        Args:
            possible_action_arr:        Tensor of actions.
            possible_predicted_q_arr:             Tensor of Q-Values.
            possible_reward_value_arr:       Tensor of reward values.
            possible_meta_data_arr:          Meta data of the actions.

        Retruns:
            Tuple(`np.ndarray` of action., Q-Value)
        '''
        raise NotImplementedError("This method must be implemented.")

    def update_q(self, reward_value_arr, next_max_q_arr):
        '''
        Update Q.
        
        Args:
            reward_value_arr:   `np.ndarray` of reward values.
            next_max_q_arr:     `np.ndarray` of maximum Q-Values in next time step.
        
        Returns:
            `np.ndarray` of real Q-Values.
        '''
        reward_value_arr = reward_value_arr.reshape((reward_value_arr.shape[0], 1))
        next_max_q_arr = next_max_q_arr.reshape((next_max_q_arr.shape[0], 1))
        # Update Q-Value.
        return self.alpha_value * (reward_value_arr + (self.gamma_value * next_max_q_arr))

    __alpha_value = 1.0

    def get_alpha_value(self):
        '''
        getter
        Learning rate.
        '''
        if isinstance(self.__alpha_value, float) is False:
            raise TypeError("The type of __alpha_value must be float.")
        return self.__alpha_value

    def set_alpha_value(self, value):
        '''
        setter
        Learning rate.
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __alpha_value must be float.")
        self.__alpha_value = value

    alpha_value = property(get_alpha_value, set_alpha_value)

    # Gamma Value.
    __gamma_value = 0.5

    def get_gamma_value(self):
        '''
        getter
        Gamma value.
        '''
        if isinstance(self.__gamma_value, float) is False:
            raise TypeError("The type of __gamma_value must be float.")
        return self.__gamma_value

    def set_gamma_value(self, value):
        '''
        setter
        Gamma value.
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __gamma_value must be float.")
        self.__gamma_value = value

    gamma_value = property(get_gamma_value, set_gamma_value)

    __q_logs_arr = None

    def get_q_logs_arr(self):
        ''' getter '''
        return self.__q_logs_arr
    
    def set_q_logs_arr(self, values):
        ''' setter '''
        raise TypeError("The `q_logs_arr` must be read-only.")
    
    q_logs_arr = property(get_q_logs_arr, set_q_logs_arr)

    # is-a `PolicySampler`.
    __policy_sampler = None

    def get_policy_sampler(self):
        ''' getter for `PolicySampler` '''
        return self.__policy_sampler

    def set_policy_sampler(self, value):
        ''' setter for `PolicySampler` '''
        if isinstance(value, PolicySampler) is False:
            raise TypeError()
        self.__policy_sampler = value

    policy_sampler = property(get_policy_sampler, set_policy_sampler)

    def get_computable_loss(self):
        ''' getter for `ComputableLoss`. '''
        return self.__computable_loss
    
    def set_computable_loss(self, value):
        ''' setter for `ComputableLoss`. '''
        if isinstance(value, ComputableLoss) is False and isinstance(value, gluon.loss.Loss) is False:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss` or `gluon.loss.Loss`.")
        self.__computable_loss = value

    computable_loss = property(get_computable_loss, set_computable_loss)
