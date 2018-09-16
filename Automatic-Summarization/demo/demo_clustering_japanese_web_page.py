# -*- coding: utf-8 -*-
from pysummarization.nlp_base import NlpBase
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.readablewebpdf.web_pdf_reading import WebPDFReading
from pysummarization.similarityfilter.encoder_decoder_clustering import EncoderDecoderClustering
from pysummarization.similarityfilter.lstm_rtrbm_clustering import LSTMRTRBMClustering
import numpy as np


def Main(url, similarity_mode="TfIdfCosine", cluster_num=10):
    '''
    Entry Point.
    
    Args:
        url:    PDF url.
    '''
    # The object of Web-scraping.
    web_scrape = WebScraping()
    # Set the object of reading PDF files.
    web_scrape.readable_web_pdf = WebPDFReading()
    # Execute Web-scraping.
    document = web_scrape.scrape(url)

    if similarity_mode == "EncoderDecoderClustering":
        # The object of `Similarity Filter`.
        # The similarity is observed by checking whether each sentence belonging to the same cluster, 
        # and if so, the similarity is `1.0`, if not, the value is `0.0`.
        # The data clustering algorithm is based on K-Means method, 
        # learning data which is embedded in hidden layer of LSTM.
        similarity_filter = EncoderDecoderClustering(
            document,
            hidden_neuron_count=200,
            epochs=100,
            batch_size=100,
            learning_rate=1e-05,
            learning_attenuate_rate=0.1,
            attenuate_epoch=50,
            bptt_tau=8,
            weight_limit=0.5,
            dropout_rate=0.5,
            test_size_rate=0.3,
            cluster_num=cluster_num,
            max_iter=100,
            debug_mode=True
        )

    elif similarity_mode == "LSTMRTRBMClustering":
        # The object of `Similarity Filter`.
        # The similarity is observed by checking whether each sentence belonging to the same cluster, 
        # and if so, the similarity is `1.0`, if not, the value is `0.0`.
        # The data clustering algorithm is based on K-Means method, 
        # learning data which is embedded in hidden layer of LSTM-RTRBM.
        similarity_filter = LSTMRTRBMClustering(
            document,
            tokenizable_doc=None,
            hidden_neuron_count=1000,
            batch_size=100,
            learning_rate=1e-03,
            seq_len=5,
            cluster_num=cluster_num,
            max_iter=100,
            debug_mode=True
        )

    else:
        raise ValueError()


    print("#" * 100)
    for i in range(cluster_num):
        print("Label: " + str(i))
        key_arr = np.where(similarity_filter.labeled_arr == i)[0]
        sentence_list = np.array(similarity_filter.sentence_list)[key_arr].tolist()
        for j in range(len(sentence_list)):
            print("".join(sentence_list[j]))
        print()

if __name__ == "__main__":
    import sys
    # PDF url.
    url = sys.argv[1]
    if len(sys.argv) > 2:
        similarity_mode = sys.argv[2]
    else:
        similarity_mode = "TfIdfCosine"
    if len(sys.argv) > 3:
        cluseter_num = int(sys.argv[3])
    else:
        cluseter_num = 10

    Main(url, similarity_mode, cluseter_num)
