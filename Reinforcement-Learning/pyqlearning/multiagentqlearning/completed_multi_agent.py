# -*- coding: utf-8 -*-
from pyqlearning.multi_agent_q_learning import MultiAgentQLearning


class CompletedMultiAgent(MultiAgentQLearning):
    '''
    Multi-Agent which learn in complete information, 
    to play a game in which knowledge about other players is available to all participants.

    Each agent can search optimal policy, observing accurate state of other agents.
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
        end_flag = False
        state_key_list = [None] * len(self.q_learning_list)
        action_key_list = [None] * len(self.q_learning_list)
        next_action_key_list = [None] * len(self.q_learning_list)
        for game in range(game_n):
            state_key = initial_state_key
            self.t = 1
            while self.t <= limit:
                for i in range(len(self.q_learning_list)):
                    state_key_list[i] = state_key
                    if game + 1 == game_n:
                        self.state_key_list.append(tuple(i, state_key_list))
                    self.q_learning_list[i].t = self.t
                    next_action_list = self.q_learning_list[i].extract_possible_actions(tuple(i, state_key_list))
                    if len(next_action_list):
                        action_key = self.q_learning_list[i].select_action(
                            state_key=tuple(i, state_key_list),
                            next_action_list=next_action_list
                        )
                        action_key_list[i] = action_key
                        reward_value = self.q_learning_list[i].observe_reward_value(
                            tuple(i, state_key_list), 
                            tuple(i, action_key_list)
                        )

                        # Check.
                        if self.q_learning_list[i].check_the_end_flag(tuple(i, state_key_list)) is True:
                            end_flag = True

                        # Max-Q-Value in next action time.
                        next_next_action_list = self.q_learning_list[i].extract_possible_actions(
                            tuple(i, action_key_list)
                        )
                        if len(next_next_action_list):
                            next_action_key = self.q_learning_list[i].predict_next_action(
                                tuple(i, action_key_list), 
                                next_next_action_list
                            )
                            next_action_key_list[i] = next_action_key
                            next_max_q = self.q_learning_list[i].extract_q_df(
                                tuple(i, action_key_list), 
                                next_action_key
                            )

                            # Update Q-Value.
                            self.q_learning_list[i].update_q(
                                state_key=tuple(i, state_key_list),
                                action_key=tuple(i, action_key_list),
                                reward_value=reward_value,
                                next_max_q=next_max_q
                            )

                            # Update State.
                            state_key = self.q_learning_list[i].update_state(
                                state_key=tuple(i, state_key_list),
                                action_key=tuple(i, action_key_list)
                            )
                            state_key_list[i] = state_key

                    # Epsode.
                    self.t += 1
                    self.q_learning_list[i].t = self.t
                    if end_flag is True:
                        break
