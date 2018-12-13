# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import random


class QLearning(metaclass=ABCMeta):
    '''
    Abstract base class and `Template Method Pattern` of Q-Learning.

    According to the Reinforcement Learning problem settings, Q-Learning 
    is a kind of Temporal Difference learning(TD Learning) that can be 
    considered as hybrid of Monte Carlo method and Dynamic Programming method. 
    As Monte Carlo method, TD Learning algorithm can learn by experience 
    without model of environment. And this learning algorithm is functional 
    extension of bootstrap method as Dynamic Programming Method.

    In this library, Q-Learning can be distinguished into Epsilon Greedy 
    Q-Leanring and Boltzmann Q-Learning. These algorithm is functionally equivalent 
    but their structures should be conceptually distinguished.

    Considering many variable parts and functional extensions in the Q-learning paradigm 
    from perspective of commonality/variability analysis in order to practice 
    object-oriented design, this abstract class defines the skeleton of a Q-Learning 
    algorithm in an operation, deferring some steps in concrete variant algorithms 
    such as Epsilon Greedy Q-Leanring and Boltzmann Q-Learning to client subclasses.
    This abstract class in this library lets subclasses redefine certain steps of 
    a Q-Learning algorithm without changing the algorithm’s structure.

    References:
        - Agrawal, S., & Goyal, N. (2011). Analysis of Thompson sampling for the multi-armed bandit problem. arXiv preprint arXiv:1111.1797.
        - Bubeck, S., & Cesa-Bianchi, N. (2012). Regret analysis of stochastic and nonstochastic multi-armed bandit problems. arXiv preprint arXiv:1204.5721.
        - Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).
        - Du, K. L., & Swamy, M. N. S. (2016). Search and optimization by metaheuristics (p. 434). New York City: Springer.
        - Kaufmann, E., Cappe, O., & Garivier, A. (2012). On Bayesian upper confidence bounds for bandit problems. In International Conference on Artificial Intelligence and Statistics (pp. 592-600).
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
        - Richard Sutton and Andrew Barto (1998). Reinforcement Learning. MIT Press.
        - Watkins, C. J. C. H. (1989). Learning from delayed rewards (Doctoral dissertation, University of Cambridge).
        - Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
        - White, J. (2012). Bandit algorithms for website optimization. ” O’Reilly Media, Inc.”.
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
    __q_df = None

    def get_q_df(self):
        '''
        getter
        '''
        if isinstance(self.__q_df, pd.DataFrame) is False and self.__q_df is not None:
            raise TypeError("The type of `__q_df` must be `pd.DataFrame`.")
        return self.__q_df

    def set_q_df(self, value):
        '''
        setter
        '''
        if isinstance(value, pd.DataFrame) is False and value is not None:
            raise TypeError("The type of `__q_df` must be `pd.DataFrame`.")
        self.__q_df = value

    q_df = property(get_q_df, set_q_df)

    def extract_q_df(self, state_key, action_key):
        '''
        Extract Q-Value from `self.q_df`.

        Args:
            state_key:      The key of state.
            action_key:     The key of action.

        Returns:
            Q-Value.

        '''
        q = 0.0
        if self.q_df is None:
            self.save_q_df(state_key, action_key, q)
            return q

        q_df = self.q_df[self.q_df.state_key == state_key]
        q_df = q_df[q_df.action_key == action_key]

        if q_df.shape[0]:
            q = float(q_df["q_value"])
        else:
            self.save_q_df(state_key, action_key, q)
        return q

    def save_q_df(self, state_key, action_key, q_value):
        '''
        Insert or update Q-Value in `self.q_df`.

        Args:
            state_key:      State.
            action_key:     Action.
            q_value:        Q-Value.

        Exceptions:
            TypeError:      If the type of `q_value` is not float.

        '''
        if isinstance(q_value, float) is False:
            raise TypeError("The type of q_value must be float.")

        new_q_df = pd.DataFrame([(state_key, action_key, q_value)], columns=["state_key", "action_key", "q_value"])
        if self.q_df is not None:
            self.q_df = pd.concat([new_q_df, self.q_df])
            self.q_df = self.q_df.drop_duplicates(["state_key", "action_key"])
        else:
            self.q_df = new_q_df
        
    # R(state)
    __r_df = None

    def get_r_df(self):
        ''' getter '''
        if isinstance(self.__r_df, pd.DataFrame) is False and self.__r_df is not None:
            raise TypeError("The type of `__r_df` must be `pd.DataFrame`.")
        return self.__r_df

    def set_r_df(self, value):
        ''' setter '''
        if isinstance(value, pd.DataFrame) is False and self.__r_df is not None:
            raise TypeError("The type of `__r_df` must be `pd.DataFrame`.")
        self.__r_df = value

    r_df = property(get_r_df, set_r_df)

    def extract_r_df(self, state_key, r_value, action_key=None):
        '''
        Insert or update R-Value in `self.r_df`.

        Args:
            state_key:     The key of state.
            r_value:       R-Value(Reward).
            action_key:    The key of action if it is nesesary for the parametar of value function.

        Exceptions:
            TypeError:      If the type of `r_value` is not float.
        '''
        if isinstance(r_value, float) is False:
            raise TypeError("The type of r_value must be float.")

        r = 0.0
        if self.r_df is None:
            self.save_r_df(state_key, r, action_key)
            return r

        r_df = self.r_df[self.r_df.state_key == state_key]
        if action_key is not None:
            r_df = r_df[r_df.action_key == action_key]
        if r_df.shape[0]:
            r = float(r_df["r_value"])
        else:
            self.save_r_df(state_key, r, action_key)
        return r

    def save_r_df(self, state_key, r_value, action_key=None):
        '''
        Insert or update R-Value in `self.r_df`.

        Args:
            state_key:     The key of state.
            r_value:       R-Value(Reward).
            action_key:    The key of action if it is nesesary for the parametar of value function.

        Exceptions:
            TypeError:      If the type of `r_value` is not float.
        '''
        if action_key is not None:
            add_r_df = pd.DataFrame([(state_key, action_key, r_value)], columns=["state_key", "action_key", "r_value"])
        else:
            add_r_df = pd.DataFrame([(state_key, r_value)], columns=["state_key", "r_value"])

        if self.r_df is not None:
            self.r_df = pd.concat([add_r_df, self.r_df])
            if action_key is not None:
                self.r_df = self.r_df.drop_duplicates(["state_key", "action_key"])
            else:
                self.r_df = self.r_df.drop_duplicates(["state_key"])
        else:
            self.r_df = add_r_df

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

    def learn(self, state_key, limit=1000):
        '''
        Learning and searching the optimal solution.
        
        Args:
            state_key:      Initial state.
            limit:          The maximum number of iterative updates based on value iteration algorithms.
        '''
        self.t = 1
        while self.t <= limit:
            next_action_list = self.extract_possible_actions(state_key)
            if len(next_action_list):
                action_key = self.select_action(
                    state_key=state_key,
                    next_action_list=next_action_list
                )
                reward_value = self.observe_reward_value(state_key, action_key)

            if len(next_action_list):
                # Max-Q-Value in next action time.
                next_state_key = self.update_state(
                    state_key=state_key,
                    action_key=action_key
                )

                next_next_action_list = self.extract_possible_actions(next_state_key)
                next_action_key = self.predict_next_action(next_state_key, next_next_action_list)
                next_max_q = self.extract_q_df(next_state_key, next_action_key)

                # Update Q-Value.
                self.update_q(
                    state_key=state_key,
                    action_key=action_key,
                    reward_value=reward_value,
                    next_max_q=next_max_q
                )
                # Update State.
                state_key = next_state_key

            # Normalize.
            self.normalize_q_value()
            self.normalize_r_value()

            # Vis.
            self.visualize_learning_result(state_key)
            # Check.
            if self.check_the_end_flag(state_key) is True:
                break

            # Epsode.
            self.t += 1

    @abstractmethod
    def select_action(self, state_key, next_action_list):
        '''
        Select action by Q(state, action).

        Abstract method for concreate usecases.

        Args:
            state_key:              The key of state.
            next_action_list:       The possible action in `self.t+1`.
                                    If the length of this list is zero, all action should be possible.

        Returns:
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
            `list` of the possible actions in `self.t+1`.

        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def observe_reward_value(self, state_key, action_key):
        '''
        Compute the reward value.
        
        Args:
            state_key:              The key of state.
            action_key:             The key of action.
        
        Returns:
            Reward value.
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
        q = self.extract_q_df(state_key, action_key)
        # Update Q-Value.
        new_q = q + self.alpha_value * (reward_value + (self.gamma_value * next_max_q) - q)
        # Save updated Q-Value.
        self.save_q_df(state_key, action_key, new_q)

    def predict_next_action(self, state_key, next_action_list):
        '''
        Predict next action by Q-Learning.

        Args:
            state_key:          The key of state in `self.t+1`.
            next_action_list:   The possible action in `self.t+1`.

        Returns:
            The key of action.

        '''
        if self.q_df is not None:
            next_action_q_df = self.q_df[self.q_df.state_key == state_key]
            next_action_q_df = next_action_q_df[next_action_q_df.action_key.isin(next_action_list)]
            if next_action_q_df.shape[0] == 0:
                return random.choice(next_action_list)
            else:
                if next_action_q_df.shape[0] == 1:
                    max_q_action = next_action_q_df["action_key"].values[0]
                else:
                    next_action_q_df = next_action_q_df.sort_values(by=["q_value"], ascending=False)
                    max_q_action = next_action_q_df.iloc[0, :]["action_key"]
                return max_q_action
        else:
            return random.choice(next_action_list)

    def update_state(self, state_key, action_key):
        '''
        Update state.
        
        This method can be overrided for concreate usecases.

        Args:
            state_key:    The key of state in `self.t`.
            action_key:   The key of action in `self.t`.
        
        Returns:
            The key of state in `self.t+1`.
        '''
        return action_key

    def normalize_q_value(self):
        '''
        Normalize q-value.
        This method should be overrided for concreate usecases.
        
        This method is called in each learning steps.
        
        For example:
            self.q_df.q_value = self.q_df.q_value / self.q_df.q_value.sum()
        '''
        pass

    def normalize_r_value(self):
        '''
        Normalize r-value.
        This method should be overrided for concreate usecases.

        This method is called in each learning steps.

        For example:
            self.r_df.r_value = self.r_df.r_value / self.r_df.r_value.sum()
        '''
        pass

    def check_the_end_flag(self, state_key):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_key:    The key of state in `self.t`.

        Returns:
            bool
        '''
        # As a rule, the learning can not be stopped.
        return False

    def visualize_learning_result(self, state_key):
        '''
        Visualize learning result.
        This method should be overrided for concreate usecases.

        This method is called in last learning steps.

        Args:
            state_key:    The key of state in `self.t`.
        '''
        pass
