# -*- coding: utf-8 -*-
from pysummarization.n_gram import Ngram
from pysummarization.nlpbase.autoabstractor.n_gram_auto_abstractor import NgramAutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.abstractabledoc.std_abstractor import StdAbstractor
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor


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
    # The object of automatic summarization with N-gram.
    auto_abstractor = NgramAutoAbstractor()
    # n-gram object
    auto_abstractor.n_gram = Ngram()
    # n of n-gram
    auto_abstractor.n = 3
    # Set tokenizer. This is japanese tokenizer with MeCab.
    auto_abstractor.tokenizable_doc = MeCabTokenizer()
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    # Execute summarization.
    result_list = auto_abstractor.summarize(document, abstractable_doc)

    # Output 3 summarized sentences.
    limit = 3
    i = 1
    for sentence in result_list["summarize_result"]:
        print(sentence)
        if i >= limit:
            break
        i += 1

if __name__ == "__main__":
    import sys
    # web site url.
    url = sys.argv[1]
    Main(url)
