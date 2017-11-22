# Reinforcement Learning

These Python Scripts create a *template method pattern* for implementing a Q-learning.

## Description

Considering many variable parts and functional extensions in the Q-learning paradigm, I implemented these Python Scripts for  demonstrations of *commonality/variability analysis* in order to design the models.

### Demonstration: Q-Learning, loosely coupled with Deep Boltzmann Machine.

[demo_maze_deep_boltzmann_q_learning.py](demo_maze_deep_boltzmann_q_learning.py) is a demonstration of how *Q-Learning* can be to *deepen*. A so-called *Deep Q-Network* (DQN) is meant only as an example. In this demonstration, let me cite the *Q-Learning* , loosely coupled with **Deep Boltzmann Machine**.

[demo_maze_deep_boltzmann_q_learning.py](demo_maze_deep_boltzmann_q_learning.py) is a simple maze solving algorithm. This algorithm allow the *agent* to search the goal in maze by *reward value* in each point in map. 

```
[['#' '#' '#' '#' '#' '#' '#' '#' '#' '#']
 ['#' 'S'  4   8   8   4   9   6   0  '#']
 ['#'  2  26   2   5   9   0   6   6  '#']
 ['#'  2   8  38   5   8   8   1   2  '#']
 ['#'  3   6   0  49   8   3   4   9  '#']
 ['#'  9   7   4   6  55   7   0   3  '#']
 ['#'  1   8   4   8   2  69   8   2  '#']
 ['#'  1   0   2   1   7   0  76   2  '#']
 ['#'  2   8   0   1   4   7   5  'G' '#']
 ['#' '#' '#' '#' '#' '#' '#' '#' '#' '#']]
```


### More detail demos

- [Webクローラ型人工知能：キメラ・ネットワークの仕様](https://media.accel-brain.com/_chimera-network-is-web-crawling-ai/)
    - 20001 bots are running as 20001 web-crawlers and 20001 web-scrapers.

### Related Case Studies

- [Webクローラ型人工知能によるパラドックス探索暴露機能の社会進化論](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/)
    - [ケーススタディ：人工知能エージェント「キメラ・ネットワーク」](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/5/#i-8)
- [深層強化学習によるベイズ主義的な情報探索に駆動された自然言語処理の意味論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/)
    - [ケーススタディ：深層学習と強化学習による「排除された第三項」の推論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/4/#i-5)
- [ヴァーチャル・リアリティにおける動物的「身体」の蒐集を媒介としたサイボーグ的な物神崇拝](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/)
    - [ケーススタディ：「人工天使ヒューズ＝ヒストリア」](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/4/#i-6)

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
