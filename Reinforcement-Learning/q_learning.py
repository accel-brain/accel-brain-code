# -*- coding: utf-8 -*-
import random
from abc import ABCMeta, abstractmethod
from operator import itemgetter


class QLearning(metaclass=ABCMeta):
    '''
    Abstract base class and `Template Method Pattern` of Q-Learning.

    Attributes:
        alpha_value:        Learning rate.
        gamma_value:        Gammma value.
        q_dict:             Q(state, action) 
        r_dict:             R(state)
        t:                  time.

    '''

    # Learning rate.
    __alpha_value = 0.1

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

    # Gamma value.
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

    # Q(state, action)
    __q_dict = {}

    def get_q_dict(self):
        '''
        getter
        Q(state, action)
        '''
        if isinstance(self.__q_dict, dict) is False:
            raise TypeError("The type of __q_dict must be dict.")
        return self.__q_dict

    def set_q_dict(self, value):
        '''
        setter
        Q(state, action)
        '''
        if isinstance(value, dict) is False:
            raise TypeError("The type of __q_dict must be dict.")
        self.__q_dict = value

    q_dict = property(get_q_dict, set_q_dict)

    def extract_q_dict(self, state_key, action_key):
        '''
        Extract Q-Value from `self.q_dict`.

        Args:
            state_key:      The key of state.
            action_key:     The key of action.

        Returns:
            Q-Value.

        '''
        q = 0.0
        try:
            q = self.q_dict[state_key][action_key]
        except KeyError:
            self.save_q_dict(state_key, action_key, q)

        return q

    def save_q_dict(self, state_key, action_key, q_value):
        '''
        Insert or update Q-Value in `self.q_dict`.

        Args:
            state_key:      State.
            action_key:     Action.
            q_value:        Q-Value.

        Exceptions:
            TypeError:      If the type of `q_value` is not float.

        '''
        if isinstance(q_value, float) is False:
            raise TypeError("The type of q_value must be float.")

        if state_key not in self.q_dict:
            self.q_dict[state_key] = {action_key: q_value}
        else:
            self.q_dict[state_key][action_key] = q_value

    # R(state)
    __r_dict = {}

    def get_r_dict(self):
        '''
        getter
        R(state)
        '''
        if isinstance(self.__r_dict, dict) is False:
            raise TypeError("The type of __r_dict must be dict.")
        return self.__r_dict

    def set_r_dict(self, value):
        '''
        setter
        R(state)
        '''
        if isinstance(value, dict) is False:
            raise TypeError("The type of __r_dict must be dict.")
        self.__r_dict = value

    r_dict = property(get_r_dict, set_r_dict)

    def extract_r_dict(self, state_key, action_key=None):
        '''
        Extract R-Value from `self.r_dict`.

        Args:
            state_key:     The key of state.
            action_key:    The key of action.

        Returns:
            R-Value(Reward).

        '''
        try:
            if action_key is None:
                return self.r_dict[state_key]
            else:
                return self.r_dict[(state_key, action_key)]
        except KeyError:
            self.save_r_dict(state_key, 0.0, action_key)
            return self.extract_r_dict(state_key, action_key)

    def save_r_dict(self, state_key, r_value, action_key=None):
        '''
        Insert or update R-Value in `self.r_dict`.

        Args:
            state_key:     The key of state.
            r_value:       R-Value(Reward).
            action_key:    The key of action if it is nesesary for the parametar of value function.

        Exceptions:
            TypeError:      If the type of `r_value` is not float.
        '''
        if isinstance(r_value, float) is False:
            raise TypeError("The type of r_value must be float.")

        if action_key is not None:
            self.r_dict[(state_key, action_key)] = r_value
        else:
            self.r_dict[state_key] = r_value

    # Time.
    __t = 0

    def get_t(self):
        '''
        getter
        Time.
        '''
        if isinstance(self.__t, int) is False:
            raise TypeError("The type of __t must be int.")
        return self.__t

    def set_t(self, value):
        '''
        setter
        Time.
        '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __t must be int.")
        self.__t = value

    t = property(get_t, set_t)

    @abstractmethod
    def learn(self, state_key, limit):
        '''
        Learning.
        
        Abstract method for concreate usecases.

        Args:
            state_key:      State.
            limit:          The number of learning.

        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def select_action(self, state_key, next_action_list):
        '''
        Select action by Q(state, action).

        Abstract method for concreate usecases.

        Args:
            state_key:              The key of state.
            next_action_list:       The possible action in `self.t+1`.
                                    If the length of this list is zero, all action should be possible.

        Retruns:
            The key of action.

        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def extract_possible_actions(self, state_key):
        '''
        Extract the list of the possible action in `self.t+1`.

        Abstract method for concreate usecases.

        Args:
            state_key       The key of state in `self.t+1`.

        Returns:
            The possible action in `self.t+1`.

        '''
        raise NotImplementedError("This method must be implemented.")

    def update_q(self, state_key, action_key, reward_value, next_max_q):
        '''
        Update Q-Value.

        Args:
            state_key:              The key of state.
            action_key:             The key of action.
            reward_value:           R-Value(Reward).
            next_max_q:             Maximum Q-Value.

        '''
        # Now Q-Value.
        q = self.extract_q_dict(state_key, action_key)
        # Update Q-Value.
        new_q = q + self.alpha_value * (reward_value + (self.gamma_value * next_max_q) - q)
        # Save updated Q-Value.
        self.save_q_dict(state_key, action_key, new_q)

    def predict_next_action(self, state_key, next_action_list):
        '''
        Predict next action by Q-Learning.

        Args:
            state_key:          The key of state in `self.t+1`.
            next_action_list:   The possible action in `self.t+1`.

        Returns:
            The key of action.

        '''
        next_action_q_list = [(action_key, self.extract_q_dict(state_key, action_key)) for action_key in next_action_list]
        random.shuffle(next_action_q_list)
        max_q_action = max(next_action_q_list, key=itemgetter(1))[0]

        return max_q_action
