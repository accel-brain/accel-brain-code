# -*- coding: utf-8 -*-
import sys
import numpy as np
import copy
from pyqlearning.qlearning.boltzmann_q_learning import BoltzmannQLearning

from pysummarization.web_scraping import WebScraping
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.n_gram import Ngram


class AutocompletionBoltzmannQLearning(BoltzmannQLearning):
    '''
    Boltzmann Q-Learning to execute an auto-completion.
    '''

    __nlp_base = None
    __n = 2

    __state_action_list_dict = {}

    def initialize(self, n=2):
        self.__nlp_base = AutoAbstractor()
        self.__nlp_base.tokenizable_doc = MeCabTokenizer()
        self.__n_gram = Ngram()
        self.__n = n

    def pre_training(self, document):
        self.__nlp_base.tokenize(document)
        token_list = self.__nlp_base.token
        token_tuple_zip = self.__n_gram.generate_ngram_data_set(
            token_list=token_list,
            n=self.__n
        )
        [self.__setup_r_q(token_tuple[0], token_tuple[1]) for token_tuple in token_tuple_zip]

    def __setup_r_q(self, state_key, action_key):
        self.__state_action_list_dict.setdefault(state_key, [])
        self.__state_action_list_dict[state_key].append(action_key)
        self.__state_action_list_dict[state_key] = list(set(self.__state_action_list_dict[state_key]))
        q_value = self.extract_q_df(state_key, action_key)
        self.save_q_df(state_key, action_key, q_value)
        r_value = self.extract_r_df(state_key, action_key)
        r_value += 1.0
        self.save_r_df(state_key, r_value, action_key)

    def lap_extract_ngram(self, document):
        self.__nlp_base.tokenize(document)
        token_list = self.__nlp_base.token
        if len(token_list) > self.__n:
            token_tuple_zip = self.__n_gram.generate_ngram_data_set(
                token_list=token_list,
                n=self.__n
            )
            token_tuple_list = [token_tuple[1] for token_tuple in token_tuple_zip]
            return token_tuple_list[-1]
        else:
            return tuple(token_list)

    def extract_possible_actions(self, state_key):
        '''
        Concreat method.

        Args:
            state_key       The key of state. this value is point in map.

        Returns:
            [(x, y)]

        '''
        if state_key in self.__state_action_list_dict:
            return self.__state_action_list_dict[state_key]
        else:
            action_list = []
            state_key_list = [action_list.extend(self.__state_action_list_dict[k]) for k in self.__state_action_list_dict.keys() if len([s for s in state_key if s in k]) > 0]
            return action_list

    def observe_reward_value(self, state_key, action_key):
        '''
        Compute the reward value.
        
        Args:
            state_key:              The key of state.
            action_key:             The key of action.
        
        Returns:
            Reward value.

        '''
        reward_value = 0.0
        if state_key in self.__state_action_list_dict:
            if action_key in self.__state_action_list_dict[state_key]:
                reward_value = 1.0

        return reward_value


if __name__ == "__main__":
    import sys
    url = sys.argv[1]
     # Object of web scraping.

    web_scrape = WebScraping()
    # Web-scraping.
    document = web_scrape.scrape(url)

    limit = 1000
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])

    alpha_value = 0.9
    gamma_value = 0.9

    boltzmann_q_learning = AutocompletionBoltzmannQLearning()
    boltzmann_q_learning.alpha_value = alpha_value
    boltzmann_q_learning.gamma_value = gamma_value
    boltzmann_q_learning.initialize(n=2)
    boltzmann_q_learning.pre_training(document=document)

    recomend_word = ""
    while True:
        document = input(">> " + recomend_word)
        document = recomend_word + document
        state_key = boltzmann_q_learning.lap_extract_ngram(document)
        print("Input word: " + str(state_key))
        boltzmann_q_learning.learn(state_key, limit=limit)
        next_action_list = boltzmann_q_learning.extract_possible_actions(state_key)
        action_key = boltzmann_q_learning.select_action(
            state_key=state_key,
            next_action_list=next_action_list
        )
        reward_value = boltzmann_q_learning.observe_reward_value(state_key, action_key)
        q_value = boltzmann_q_learning.extract_q_dict(state_key, action_key)

        print("Predicted word: " + str(action_key))
        print("Reward: " + str(reward_value))
        print("Q-Value: " + str(q_value))

        boltzmann_q_learning.pre_training(document=document)
        recomend_word = document + "".join(list(action_key)[1:])
