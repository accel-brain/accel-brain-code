# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.nlp_base import NlpBase
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.vectorizablesentence.encoder_decoder import EncoderDecoder


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
    all_token_list = []
    for i in range(len(sentence_list)):
        nlp_base.tokenize(sentence_list[i])
        all_token_list.extend(nlp_base.token)
        sentence_list[i] = nlp_base.token
    
    vectorlizable_sentence = EncoderDecoder()
    vectorlizable_sentence.learn(
        sentence_list=sentence_list, 
        token_master_list=list(set(all_token_list))
    )
    feature_points_arr = vectorlizable_sentence.vectorlize(sentence_list[:5])
    
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

    Main(url)
