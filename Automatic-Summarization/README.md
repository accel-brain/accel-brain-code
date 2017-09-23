# Automatic Summarization

These python scripts create a summary with the major points of the original document or web-scraped text.

## Description

This is a demonstration on automatic summarization, abstraction, or filtering for my case study in the context of my website.

## Usecase: A English Web-Page Summarization

Run the batch program: [demo_summarization_english_web_page.py](https://github.com/chimera0/accel-brain-code/blob/master/Automatic-Summarization/demo_summarization_english_web_page.py)

```
python demo_summarization_english_web_page.py {URL}

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
    - [ケーススタディ：文書自動要約技術](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/4/#i-12)
    - [ケーススタディ：人工知能エージェント「キメラ・ネットワーク」](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/5/#i-8) (Japanese)

## Version
- 1.0

## Author

- chimera0

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0

## Requires

- Python3.4.4

# References

- Luhn, Hans Peter. "The automatic creation of literature abstracts." IBM Journal of research and development 2.2 (1958): 159-165.
- Matthew A. Russell　著、佐藤 敏紀、瀬戸口 光宏、原川 浩一　監訳、長尾 高弘　訳『入門 ソーシャルデータ 第2版――ソーシャルウェブのデータマイニング』 2014年06月 発行

