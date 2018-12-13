# -*- coding: utf-8 -*-
from pyqlearning.multi_agent_q_learning import MultiAgentQLearning
import copy


class AlternatedMultiAgent(MultiAgentQLearning):
    '''
    Multi-Agent which do alternated learn.
    '''

    def learn(self, initial_state_key, limit=1000, game_n=1):
        '''
        Multi-Agent Learning.
        
        Override.
        
        Args:
            initial_state_key:  Initial state.
            limit:              Limit of the number of learning.
            game_n:             The number of games.
            
        '''
        end_flag_list = [False] * len(self.q_learning_list)
        for game in range(game_n):
            state_key = copy.copy(initial_state_key)
            self.t = 1
            while self.t <= limit:
                for i in range(len(self.q_learning_list)):
                    if game + 1 == game_n:
                        self.state_key_list.append((i, copy.copy(state_key)))
                    self.q_learning_list[i].t = self.t
                    next_action_list = self.q_learning_list[i].extract_possible_actions(state_key)

                    if len(next_action_list):
                        action_key = self.q_learning_list[i].select_action(
                            state_key=state_key,
                            next_action_list=next_action_list
                        )
                        reward_value = self.q_learning_list[i].observe_reward_value(state_key, action_key)

                        # Check.
                        if self.q_learning_list[i].check_the_end_flag(state_key) is True:
                            end_flag_list[i] = True

                        # Max-Q-Value in next action time.
                        next_state_key = self.q_learning_list[i].update_state(
                            state_key=state_key,
                            action_key=action_key
                        )
                        next_next_action_list = self.q_learning_list[i].extract_possible_actions(next_state_key)
                        if len(next_next_action_list):
                            next_action_key = self.q_learning_list[i].predict_next_action(
                                next_state_key,
                                next_next_action_list
                            )
                            next_max_q = self.q_learning_list[i].extract_q_df(next_state_key, next_action_key)

                            # Update Q-Value.
                            self.q_learning_list[i].update_q(
                                state_key=state_key,
                                action_key=action_key,
                                reward_value=reward_value,
                                next_max_q=next_max_q
                            )

                        # Update State.
                        state_key = next_state_key

                    # Epsode.
                    self.t += 1
                    self.q_learning_list[i].t = self.t
                    if False not in end_flag_list:
                        break
