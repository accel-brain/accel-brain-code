# -*- coding: utf-8 -*-
from nlpbase.auto_abstractor import AutoAbstractor
from web_scraping import WebScraping
from abstractabledoc.std_abstractor import StdAbstractor
from abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor


def Main(url):
    web_scrape = WebScraping()
    document = web_scrape.scrape(url)

    auto_abstractor = AutoAbstractor()
    abstractable_doc = TopNRankAbstractor()
    result_list = auto_abstractor.summarize(document, abstractable_doc)
    limit = 3
    i = 0
    for sentence in result_list["summarize_result"]:
        print(sentence)
        if i >= limit:
            break
        i += 1

if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    Main(url)
