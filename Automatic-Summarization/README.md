# Automatic Summarization Library: pysummarization

`pysummarization` is Python3 library for the automatic summarization, document abstraction, and text filtering.

## Description

The function of this library is automatic summarization using a kind of natural language processing. This library enable you to create a summary with the major points of the original document or web-scraped text that filtered by text clustering.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Automatic-Summarization/](https://code.accel-brain.com/Automatic-Summarization/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Installation

Install using pip:

```sh
pip install pysummarization
```

### Source code

The source code is currently hosted on GitHub.

- [accel-brain-code/Automatic-Summarization](https://github.com/chimera0/accel-brain-code/tree/master/Automatic-Summarization)

### Python package index(PyPI)

Installers for the latest released version are available at the Python package index.

- [pysummarization : Python Package Index](https://pypi.python.org/pypi/pysummarization/)

### Dependencies

- numpy: v1.13.3 or higher.
- nltk: v3.2.3 or higher.
- mecab-python3: v0.7 or higher.
- pdfminer2
- pyquery:v1.2.17 or higher.

## Usecase: Summarize an English string argument.

Import Python modules.

```python
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
```

Prepare an English string argument.

```python
document = "Natural language generation (NLG) is the natural language processing task of generating natural language from a machine representation system such as a knowledge base or a logical form. Psycholinguists prefer the term language production when such formal representations are interpreted as models for mental representations."
```

And instantiate objects and call the method.

```python
# Object of automatic summarization.
auto_abstractor = AutoAbstractor()
# Set tokenizer.
auto_abstractor.tokenizable_doc = SimpleTokenizer()
# Set delimiter for making a list of sentence.
auto_abstractor.delimiter_list = [".", "\n"]
# Object of abstracting and filtering document.
abstractable_doc = TopNRankAbstractor()
# Summarize document.
result_dict = auto_abstractor.summarize(document, abstractable_doc)

# Output result.
for sentence in result_dict["summarize_result"]:
    print(sentence)
```

The `result_dict` is a dict. this format is as follows.

```python
 dict{
     "summarize_result": "The list of summarized sentences.", 
     "scoring_data":     "The list of scores(Rank of importance)."
 }
```

## Usecase: Summarize Japanese string argument.

Import Python modules.

```python
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
```

Prepare an English string argument.

```python
document = "自然言語処理（しぜんげんごしょり、英語: natural language processing、略称：NLP）は、人間が日常的に使っている自然言語をコンピュータに処理させる一連の技術であり、人工知能と言語学の一分野である。「計算言語学」（computational linguistics）との類似もあるが、自然言語処理は工学的な視点からの言語処理をさすのに対して、計算言語学は言語学的視点を重視する手法をさす事が多い[1]。データベース内の情報を自然言語に変換したり、自然言語の文章をより形式的な（コンピュータが理解しやすい）表現に変換するといった処理が含まれる。応用例としては予測変換、IMEなどの文字変換が挙げられる。"
```

And instantiate objects and call the method.

```python
# Object of automatic summarization.
auto_abstractor = AutoAbstractor()
# Set tokenizer for Japanese.
auto_abstractor.tokenizable_doc = MeCabTokenizer()
# Set delimiter for making a list of sentence.
auto_abstractor.delimiter_list = ["。", "\n"]
# Object of abstracting and filtering document.
abstractable_doc = TopNRankAbstractor()
# Summarize document.
result_dict = auto_abstractor.summarize(document, abstractable_doc)

# Output result.
for sentence in result_dict["summarize_result"]:
    print(sentence)
```

## Usecase: English Web-Page Summarization

Run the batch program: [demo_summarization_english_web_page.py](https://github.com/chimera0/accel-brain-code/blob/master/Automatic-Summarization/demo_summarization_english_web_page.py)

```
python demo_summarization_english_web_page.py {URL}
```

- {URL}: web site URL.

### Demo

Let's summarize this page: [Natural_language_generation - Wikipedia](https://en.wikipedia.org/wiki/Natural_language_generation).

```
python demo_summarization_english_web_page.py https://en.wikipedia.org/wiki/Natural_language_generation
```

The result is as follows.
```
Natural language generation From Wikipedia, the free encyclopedia Jump to: navigation , search Natural language generation ( NLG ) is the natural language processing task of generating natural language from a machine representation system such as a knowledge base or a logical form .

 Psycholinguists prefer the term language production when such formal representations are interpreted as models for mental representations.

 It could be said an NLG system is like a translator that converts data into a natural language representation.
```

## Usecase: Japanese Web-Page Summarization

Run the batch program: [demo_summarization_japanese_web_page.py](https://github.com/chimera0/accel-brain-code/blob/master/Automatic-Summarization/demo_summarization_japanese_web_page.py)

```
python demo_summarization_japanese_web_page.py {URL}
```
- {URL}: web site URL.

### Demo

Let's summarize this page: [自動要約 - Wikipedia](https://ja.wikipedia.org/wiki/%E8%87%AA%E5%8B%95%E8%A6%81%E7%B4%84).

```
python demo_summarization_japanese_web_page.py https://ja.wikipedia.org/wiki/%E8%87%AA%E5%8B%95%E8%A6%81%E7%B4%84
```

The result is as follows.
```
 自動要約 （じどうようやく）は、 コンピュータプログラム を用いて、文書からその要約を作成する処理である。

自動要約の応用先の1つは Google などの 検索エンジン であるが、もちろん独立した1つの要約プログラムといったものもありうる。

 単一文書要約と複数文書要約 [ 編集 ] 単一文書要約 は、単一の文書を要約の対象とするものである。

例えば、1つの新聞記事を要約する作業は単一文書要約である。
```

## Usecase: Japanese Web-Page Summarization with N-gram

The minimum unit of token is not necessarily `a word` in automatic summarization. `N-gram` is also applicable to the tokenization.

Run the batch program: [demo_with_n_gram_japanese_web_page.py](https://github.com/chimera0/accel-brain-code/blob/master/Automatic-Summarization/demo_with_n_gram_japanese_web_page.py)

```
python demo_with_n_gram_japanese_web_page.py {URL}
```
- {URL}: web site URL.

### Demo

Let's summarize this page:[情報検索 - Wikipedia](https://ja.wikipedia.org/wiki/%E6%83%85%E5%A0%B1%E6%A4%9C%E7%B4%A2).

```
python demo_with_n_gram_japanese_web_page.py https://ja.wikipedia.org/wiki/%E6%83%85%E5%A0%B1%E6%A4%9C%E7%B4%A2
```

The result is as follows.

```
情報検索アルゴリズムの詳細については 情報検索アルゴリズム を参照のこと。

 パターンマッチング 検索質問として入力された表現をそのまま含む文書を検索するアルゴリズム。

 ベクトル空間モデル キーワード等を各 次元 として設定した高次元 ベクトル空間 を想定し、検索の対象とするデータやユーザによる検索質問に何らかの加工を行い ベクトル を生成する
```

## Usecase: Japanese Web-Page Summarization with Similarity Filter

If the sentences you want to summarize consist of repetition of same or similar sense in different words, the summary results may also be redundant. Then before summarization, you should filter the mutually similar, tautological, pleonastic, or redundant sentences to extract features having an information quantity. The function of `SimilarityFilter` is to cut-off the sentences having the state of resembling or being alike by calculating the similarity measure.

### Methods for calculating the similarity measure

There are some methods for calculating the similarity measure. In this library, Dice coefficient, Jaccard coefficient, and Simpson coefficient between two sentences is calculated as follows.

Import Python modules for calculating the similarity measure and instantiate the object.

```python
from pysummarization.similarityfilter.dice import Dice
similarity_filter = Dice()
```

or

```python
from pysummarization.similarityfilter.jaccard import Jaccard
similarity_filter = Jaccard()
```

or

```python
from pysummarization.similarityfilter.simpson import Simpson
similarity_filter = Simpson()
```

If you want to calculate similarity between two sentences, call `calculate` method as follow.

```python
# Instantiate
similarity_filter = Jaccard()
# Tokenized sentences
token_list_x = ["Dice", "coefficient", "is", "a", "similarity", "measure", "."]
token_list_y = ["Jaccard", "coefficient", "is", "a", "similarity", "measure", "."]
# 0.75
similarity_num = similarity_filter.calculate(token_list_x, token_list_y)
```

### Tf-Idf and Cosine similarity

If you want to calculate similarity with Tf-Idf cosine similarity, instantiate `TfIdfCosine`.

```python
from pysummarization.similarityfilter.tfidf_cosine import TfIdfCosine
similarity_filter = TfIdfCosine()
```

### Filtering similar sentences and summarization

The function of these methods is to cut-off mutually similar sentences. In text summarization, basic usage of this function is as follow. After all, `SimilarityFilter` is delegated as well as GoF's Strategy Pattern.

Import Python modules for NLP and text summarization.

```python
from pysummarization.nlp_base import NlpBase
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from pysummarization.similarityfilter.tfidf_cosine import TfIdfCosine
```

Instantiate object of the NLP.

```python
# The object of the NLP.
nlp_base = NlpBase()
# Set tokenizer. This is japanese tokenizer with MeCab.
nlp_base.tokenizable_doc = MeCabTokenizer()
```

Instantiate object of `SimilarityFilter` and set the cut-off threshold.

```python
# The object of `Similarity Filter`. 
# The similarity observed by this object is so-called cosine similarity of Tf-Idf vectors.
similarity_filter = TfIdfCosine()

# Set the object of NLP.
similarity_filter.nlp_base = nlp_base

# If the similarity exceeds this value, the sentence will be cut off.
similarity_filter.similarity_limit = 0.25
```

Prepare sentences you want to summarize.

```python
# Summarized sentences (sited from http://ja.uncyclopedia.info/wiki/%E5%86%97%E8%AA%9E%E6%B3%95).
document = "冗語法（じょうごほう、レデュンダンシー、redundancy、jōgohō）とは、何度も何度も繰り返し重ねて重複して前述されたのと同じ意味の同様である同意義の文章を、必要あるいは説明か理解を要求された以上か、伝え伝達したいと意図された、あるいは表し表現したい意味以上に、繰り返し重ねて重複して繰り返すことによる、不必要であるか、または余分な余計である文章の、必要以上の使用であり、何度も何度も繰り返し重ねて重複して前述されたのと同じ意味の同様の文章を、必要あるいは説明か理解を要求された以上か、伝え伝達したいと意図された、あるいは表し表現したい意味以上に、繰り返し重ねて重複して繰り返すことによる、不必要であるか、または余分な文章の、必要以上の使用である。これが冗語法（じょうごほう、レデュンダンシー、redundancy、jōgohō）である。基本的に、冗語法（じょうごほう、レデュンダンシー、redundancy、jōgohō）が多くの場合において概して一般的に繰り返される通常の場合は、普通、同じ同様の発想や思考や概念や物事を表し表現する別々の異なった文章や単語や言葉が何回も何度も余分に繰り返され、その結果として発言者の考えが何回も何度も言い直され、事実上、実際に同じ同様の発言が何回も何度にもわたり、幾重にも言い換えられ、かつ、同じことが何回も何度も繰り返し重複して過剰に回数を重ね前述されたのと同じ意味の同様の文章が何度も何度も不必要に繰り返される。通常の場合、多くの場合において概して一般的にこのように冗語法（じょうごほう、レデュンダンシー、redundancy、jōgohō）が繰り返される。"
```

Instantiate object of `AutoAbstractor` and call the method.

```python
# The object of automatic sumamrization.
auto_abstractor = AutoAbstractor()
# Set tokenizer. This is japanese tokenizer with MeCab.
auto_abstractor.tokenizable_doc = MeCabTokenizer()
# Object of abstracting and filtering document.
abstractable_doc = TopNRankAbstractor()
# Delegate the objects and execute summarization.
result_dict = auto_abstractor.summarize(document, abstractable_doc, similarity_filter)
```

### Demo

Let's summarize this page:[循環論法 - Wikipedia](https://ja.wikipedia.org/wiki/%E5%BE%AA%E7%92%B0%E8%AB%96%E6%B3%95).

Run the batch program: [demo_similarity_filtering_japanese_web_page.py](https://github.com/chimera0/accel-brain-code/blob/master/Automatic-Summarization/demo_similarity_filtering_japanese_web_page.py)

```
python demo_similarity_filtering_japanese_web_page.py {URL} {SimilarityFilter} {SimilarityLimit}
```
- {URL}: web site URL.
- {SimilarityFilter}: The object of `SimilarityFilter`: `Dice`, `Jaccard`, `Simpson`, or `TfIdfCosine`.
- {SimilarityLimit}: The cut-off threshold.

For instance, command line argument is as follows:

```
python demo_similarity_filtering_japanese_web_page.py https://ja.wikipedia.org/wiki/%E5%BE%AA%E7%92%B0%E8%AB%96%E6%B3%95 Jaccard 0.3
```

The result is as follows.

```
循環論法 出典: フリー百科事典『ウィキペディア（Wikipedia）』 移動先: 案内 、 検索 循環論法 （じゅんかんろんぽう、circular reasoning, circular logic, vicious circle [1] ）とは、 ある命題の 証明 において、その命題を仮定した議論を用いること [1] 。

証明すべき結論を前提として用いる論法 [2] 。

 ある用語の 定義 を与える表現の中にその用語自体が本質的に登場していること [1]
```

## More detail demos

- [Webクローラ型人工知能：キメラ・ネットワークの仕様](https://media.accel-brain.com/_chimera-network-is-web-crawling-ai/) (Japanese)
    - 20001 bots are running as 20001 web-crawlers and 20001 web-scrapers.
- [「代理演算」一覧 | Welcome to Singularity](https://media.accel-brain.com/category/agency-operation/) (Japanese)
    - 20001 bots are running as 20001 blogers and 20001 "content curation writers".

## Related PoC

- [Webクローラ型人工知能によるパラドックス探索暴露機能の社会進化論](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/) (Japanese)
    - [プロトタイプの開発：文書自動要約技術](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/4/#i-12)
    - [プロトタイプの開発：人工知能エージェント「キメラ・ネットワーク」](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/5/#i-8)

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0

# References

- Luhn, Hans Peter. "The automatic creation of literature abstracts." IBM Journal of research and development 2.2 (1958): 159-165.
- Matthew A. Russell　著、佐藤 敏紀、瀬戸口 光宏、原川 浩一　監訳、長尾 高弘　訳『入門 ソーシャルデータ 第2版――ソーシャルウェブのデータマイニング』 2014年06月 発行
