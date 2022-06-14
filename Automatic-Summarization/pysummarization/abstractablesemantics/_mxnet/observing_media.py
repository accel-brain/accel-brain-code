# -*- coding: utf-8 -*-
from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.iteratabledata._mxnet.gauss_transformer_iterator import GaussTransformerIterator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.controllablemodel._mxnet.transformer_controller import TransformerController
from accelbrainbase.controllablemodel._mxnet.transforming_auto_encoder_controller import TransformingAutoEncoderController
from pysummarization.iteratabledata._mxnet.transformeriterator.token_transforming_auto_encoder_iterator import TokenTransformingAutoEncoderIterator
from pysummarization.iteratabledata._mxnet.transformeriterator.bi_sentence_transformer_iterator import BiSentenceTransformerIterator
from pysummarization.vectorizabletoken.t_hot_vectorizer import THotVectorizer

from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd


class ObservingMedia(object):
    '''
    Observing media.
    '''

    def __init__(
        self,
        document,
        tokenizable_doc,
        vectorizable_token=None,
        stop_words_list=["*"],
        epochs=1000,
        batch_size=20,
        head_n=4,
        seq_len=2,
        ctx=mx.gpu(),
        hidden_dim=128,
        test_size=0.3,
        learning_rate=0.000001,
        layer_n=3,
        dropout_rate=0.5,
    ):
        """
        Init.

        Args:
            document:       `str` of target document.
            tokenizable_doc:    is-a `TokenizableDoc`.
            vectorizable_token: is-a `VectorizableToken`.
            stop_words_list:    `list` of stop words.
            epochs:             `int` of epochs.
            batch_size:         `int` of batch size.
            head_n:             `int` of the number of heads for multi-head attention.
            seq_len:            `int` of the length of sequences.
            ctx:                `mx.gpu()` or `mx.cpu()`.
            hidden_dim:         `int` of the number of units in hidden layer(encoder and decoder, not reconstructor).
            test_size:          `float` of test dataset size.
            learning_rate:      `int` of learning rate.
            layer_n:            `int` of the number of layers.
            dropout_rate:       `float` of dropout rate.
        """
        #sentence_list = document.split("\n")
        sentence_list = document.splitlines()

        token_list = []
        seq_token_list = []
        for sentence in sentence_list:
            _token_list = tokenizable_doc.tokenize(sentence)
            _token_list = [v.replace(" ", "") for v in _token_list if v not in stop_words_list]
            [seq_token_list.append(_token) for _token in _token_list]
            [token_list.append(v) for v in _token_list]

        if vectorizable_token is None:
            vectorizable_token = THotVectorizer(token_list)

        transformer_iterator = TokenTransformingAutoEncoderIterator(
            seq_token_list=seq_token_list,
            vectorizable_token=vectorizable_token, 
            epochs=epochs,
            batch_size=batch_size,
            seq_len=seq_len,
            test_size=test_size,
            norm_mode=None,
            noiseable_data=GaussNoise(sigma=1e-03, mu=0.0),
            scale=1.0,
            ctx=ctx
        )
        computable_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, from_logits=False)

        transformer_controller = TransformingAutoEncoderController(
            learning_rate=learning_rate,
            layer_n=layer_n,
            head_n=head_n,
            seq_len=seq_len,
            depth_dim=vectorizable_token.dim,
            self_attention_activation_list=["gelu"],
            multi_head_attention_activation_list=["gelu"],
            fc_activation_list=["gelu"],
            hidden_dim=hidden_dim,
            optimizer_name="Adam",
            learning_attenuate_rate=1.0,
            attenuate_epoch=50,
            hybridize_flag=True,
            dropout_rate=dropout_rate,
        )
        self.transformer_iterator = transformer_iterator
        self.transformer_controller = transformer_controller
    
    def learn(self):
        '''
        Learning.
        '''
        self.transformer_controller.learn(self.transformer_iterator)

    def extract_media(self, test_mode=True):
        '''
        Extracting tokens of media.

        Args:
            test_mode:      `bool`. If `True`, this method extracts from test dataset.
        
        Returns:
            `pd.DataFrame`.
        '''
        all_data_arr = None
        for encoded_observed_arr, decoded_observed_arr, encoded_mask_arr, decoded_mask_arr, token_list in self.transformer_iterator.generate_samples_and_noises(
            test_mode=test_mode
        ):
            decoded_arr = self.transformer_controller.inference(
                encoded_observed_arr, 
                decoded_observed_arr, 
                encoded_mask_arr=encoded_mask_arr,
                decoded_mask_arr=decoded_mask_arr,
            )
            loss_arr = self.transformer_controller.compute_loss(
                pred_arr=decoded_arr, 
                labeled_arr=decoded_observed_arr
            )
            loss_arr = loss_arr.asnumpy()
            token_arr = np.array(token_list).reshape(-1, 1)
            data_arr = np.c_[loss_arr[:token_arr.shape[0]], token_arr]
            if all_data_arr is None:
                all_data_arr = data_arr
            else:
                all_data_arr = np.r_[all_data_arr, data_arr]

        media_df = pd.DataFrame(all_data_arr.tolist(), columns=["loss", "token"]).sort_values(by=["loss"])

        media_df.token = media_df.token.apply(self.__switch_sep)
        media_df = media_df.drop_duplicates(["token"])
        media_df.loss = media_df.loss.astype(float)
        media_df = media_df.sort_values(by=["loss"])
        return media_df

    def extract_form(self, media_token_list, test_mode=True):
        '''
        Extracting tokens of forms.

        Args:
            media_token_list:   `list` of `str` of tokens of media.
            test_mode:      `bool`. If `True`, this method extracts from test dataset.
        
        Returns:
            `pd.DataFrame`.
        '''

        all_data_arr = None
        for encoded_observed_arr, decoded_observed_arr, encoded_mask_arr, decoded_mask_arr, token_list in self.transformer_iterator.generate_inferenced_samples(
            media_token_list, 
            test_mode=test_mode
        ):
            decoded_arr = self.transformer_controller.inference(
                encoded_observed_arr, 
                decoded_observed_arr, 
                encoded_mask_arr=encoded_mask_arr,
                decoded_mask_arr=decoded_mask_arr,
            )
            loss_arr = self.transformer_controller.compute_loss(
                pred_arr=decoded_arr, 
                labeled_arr=decoded_observed_arr
            )
            loss_arr = loss_arr.asnumpy()
            token_arr = np.array(token_list).reshape(-1, 1)
            data_arr = np.c_[loss_arr[:token_arr.shape[0]], token_arr]
            if all_data_arr is None:
                all_data_arr = data_arr
            else:
                all_data_arr = np.r_[all_data_arr, data_arr]

        form_df = pd.DataFrame(all_data_arr.tolist(), columns=["loss", "token"]).sort_values(by=["loss"])
        form_df.token = form_df.token.apply(self.__switch_sep)
        form_df = form_df.drop_duplicates(["token"])
        form_df.loss = form_df.loss.astype(float)
        form_df = form_df.sort_values(by=["loss"])

        return form_df

    def __switch_sep(self, v):
        return v.replace("<sep>", ", ")


def Main(
    file_path, 
    transfer_learning_flag=False, 
    not_learning_flag=False, 
    stop_word_path=None,
    media_n=5,
    form_n=25
):
    from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
    from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

    logger = getLogger("accelbrainbase")
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    tokenizable_doc = MeCabTokenizer()
    tokenizable_doc.mecab_system_dic = "C:/Program Files/MeCab/dic/mecab-ipadic-neologd"
    tokenizable_doc.part_of_speech = ["名詞"]

    if stop_word_path is None:
        stop_words_list = [
            "これ", 
            "それ",
            "こと", 
            "さ", 
            "いわく", 
            "ところ", 
            "もの", 
            "*",
            "性",
            "的",
            "他",
            "ここ",
            "そこ",
            "あそこ",
            "それら",
            "これら",
            "彼",
            "彼女",
            "私",
            "あなた",
            "貴方",
            "彼ら",
            "彼女ら",
            "何"
        ]
    else:
        with open(stop_word_path, encoding="utf-8") as f:
            stop_words = f.read()
        
        #stop_words_list = stop_words.split("\n")
        stop_words_list = stop_words.splitlines()

    with open(file_path, encoding="utf-8") as f:
        document = f.read()

    observing_media = ObservingMedia(
        document,
        tokenizable_doc,
        vectorizable_token=None,
        stop_words_list=stop_words_list,
        epochs=1000,
        batch_size=20,
        head_n=4,
        seq_len=2,
        ctx=mx.gpu(),
        hidden_dim=128,
        test_size=0.3,
        learning_rate=0.000001,
        layer_n=3,
        dropout_rate=0.5,
    )
    if transfer_learning_flag is True:
        observing_media.transformer_controller.load_parameters(
            "transforming_auto_encoder.params", 
            ctx=mx.gpu()
        )

    if not_learning_flag is False:
        observing_media.learn()
        observing_media.transformer_controller.save_parameters("transforming_auto_encoder.params")

    test_mode = True
    media_df = observing_media.extract_media(test_mode=test_mode)
    print("Extracting media is end.")
    media_master_df = media_df.drop_duplicates(["token"])

    media_form_list = []
    for i in range(media_n):
        media_token_list = media_master_df.token.values[i].split(",")
        media_token_list = [v.replace(" ", "") for v in media_token_list]
        form_df = observing_media.extract_form(media_token_list, test_mode=test_mode)

        form_df = form_df[form_df.token != media_master_df.token.values[i]]
        for j in range(form_n):
            media_form_list.append(
                (
                    media_df.token.values[i],
                    form_df.token.values[j],
                    form_df.loss.values[j]
                )
            )

    media_form_df = pd.DataFrame(media_form_list, columns=["media", "form", "loss"])
    media_form_df.to_csv("media_form.csv", index=False)

    print("end.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Obsering media.")
    
    parser.add_argument(
        "-t",
        "--target_file_path",
        type=str,
        default="media_form_target.txt",
        help="Target document."
    )

    parser.add_argument(
        "-tl",
        "--transfer_learning_flag",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-nl",
        "--not_learning_flag",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-s",
        "--stop_word_path",
        type=str,
        default=None,
        help="Stop word list."
    )

    parser.add_argument(
        "-mn",
        "--media_n",
        type=int,
        default=5,
        help="The number of media."
    )

    parser.add_argument(
        "-fn",
        "--form_n",
        type=int,
        default=25,
        help="The number of form."
    )

    args = parser.parse_args()
    params_dict = vars(args)
    print(params_dict)

    Main(
        params_dict["target_file_path"],
        params_dict["transfer_learning_flag"],
        params_dict["not_learning_flag"],
        params_dict["stop_word_path"],
        params_dict["media_n"],
        params_dict["form_n"],
    )
