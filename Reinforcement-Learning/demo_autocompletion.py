import sys
import numpy as np
from devsample.autocompletion_boltzmann_q_learning import AutocompletionBoltzmannQLearning
from pysummarization.web_scraping import WebScraping


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
