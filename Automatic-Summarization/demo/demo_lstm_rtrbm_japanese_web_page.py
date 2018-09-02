# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.nlp_base import NlpBase
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.vectorizablesentence.lstm_rtrbm import LSTMRTRBM


def Main(url):
    '''
    Entry Point.
    
    Args:
        url:    target url.
    '''
    # The object of Web-Scraping.
    web_scrape = WebScraping()
    # Execute Web-Scraping.
    document = web_scrape.scrape(url)
    # The object of NLP.
    nlp_base = NlpBase()
    # Set tokenizer. This is japanese tokenizer with MeCab.
    nlp_base.tokenizable_doc = MeCabTokenizer()

    sentence_list = nlp_base.listup_sentence(document)

    batch_size = 10
    if len(sentence_list) < batch_size:
        raise ValueError("The number of extracted sentences is insufficient.")

    all_token_list = []
    for i in range(len(sentence_list)):
        nlp_base.tokenize(sentence_list[i])
        all_token_list.extend(nlp_base.token)
        sentence_list[i] = nlp_base.token
    
    vectorlizable_sentence = LSTMRTRBM()
    vectorlizable_sentence.learn(
        sentence_list=sentence_list, 
        token_master_list=list(set(all_token_list)),
        hidden_neuron_count=1000,
        batch_size=batch_size,
        learning_rate=1e-03,
        seq_len=5
    )
    test_list = sentence_list[:batch_size]
    feature_points_arr = vectorlizable_sentence.vectorize(test_list)

    print("Feature points (Top 5 sentences):")
    print(feature_points_arr)

if __name__ == "__main__":
    import sys
    from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

    # web site url.
    url = sys.argv[1]

    logger = getLogger("pydbm")
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger = getLogger("pysummarization")
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    Main(url)
