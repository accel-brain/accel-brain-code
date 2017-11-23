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

## Usecase: Summarize an Japanese string argument.

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

## Usecase: A English Web-Page Summarization

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

## Usecase: A Japanese Web-Page Summarization

Run the batch program: [demo_summarization_japanese_web_page.py](https://github.com/chimera0/accel-brain-code/blob/master/Automatic-Summarization/demo_summarization_japanese_web_page.py)

```
python3 demo_summarization_japanese_web_page.py {URL}
```
- {URL}: web site URL.

### Demo

Let's summarize this page: [自動要約 - Wikipedia](https://ja.wikipedia.org/wiki/%E8%87%AA%E5%8B%95%E8%A6%81%E7%B4%84).

```
python3 demo_summarization_japanese_web_page.py https://ja.wikipedia.org/wiki/%E8%87%AA%E5%8B%95%E8%A6%81%E7%B4%84
```

The result is as follows.
```
 自動要約 （じどうようやく）は、 コンピュータプログラム を用いて、文書からその要約を作成する処理である。

自動要約の応用先の1つは Google などの 検索エンジン であるが、もちろん独立した1つの要約プログラムといったものもありうる。

 単一文書要約と複数文書要約 [ 編集 ] 単一文書要約 は、単一の文書を要約の対象とするものである。

例えば、1つの新聞記事を要約する作業は単一文書要約である。
```

## Usecase: N-gram

The minimum unit of token is not necessarily `a word` in automatic summarization. `N-gram` is also applicable to the tokenization.

Run the batch program: [demo_with_n_gram_japanese_web_page.py](https://github.com/chimera0/accel-brain-code/blob/master/Automatic-Summarization/demo_with_n_gram_japanese_web_page.py)

```
python3 demo_with_n_gram_japanese_web_page.py {URL}
```
- {URL}: web site URL.

### Demo

Let's summarize this page:[情報検索 - Wikipedia](https://ja.wikipedia.org/wiki/%E6%83%85%E5%A0%B1%E6%A4%9C%E7%B4%A2).

```
python3 demo_with_n_gram_japanese_web_page.py https://ja.wikipedia.org/wiki/%E6%83%85%E5%A0%B1%E6%A4%9C%E7%B4%A2
```

The result is as follows.

```
情報検索アルゴリズムの詳細については 情報検索アルゴリズム を参照のこと。

 パターンマッチング 検索質問として入力された表現をそのまま含む文書を検索するアルゴリズム。

 ベクトル空間モデル キーワード等を各 次元 として設定した高次元 ベクトル空間 を想定し、検索の対象とするデータやユーザによる検索質問に何らかの加工を行い ベクトル を生成する
```

### More detail demos

- [Webクローラ型人工知能：キメラ・ネットワークの仕様](https://media.accel-brain.com/_chimera-network-is-web-crawling-ai/) (Japanese)
    - 20001 bots are running as 20001 web-crawlers and 20001 web-scrapers.
- [「代理演算」一覧 | Welcome to Singularity](https://media.accel-brain.com/category/agency-operation/) (Japanese)
    - 20001 bots are running as 20001 blogers and 20001 "content curation writers".

### Related Case Studies

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

