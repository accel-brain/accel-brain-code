# Reinforcement Learning Library: pyqlearning

`pyqlearning` is Python library to implement Reinforcement Learning, especially for Q-Learning.

## Description

Considering many variable parts and functional extensions in the Q-learning paradigm, I implemented these Python Scripts for  demonstrations of *commonality/variability analysis* in order to design the models.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Reinforcement-Learning/](https://code.accel-brain.com/Reinforcement-Learning/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Installation

Install using pip:

```sh
pip install pyqlearning
```

### Source code

The source code is currently hosted on GitHub.

- [accel-brain-code/Reinforcement-Learning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning)

### Python package index(PyPI)

Installers for the latest released version are available at the Python package index.

- [pyqlearning : Python Package Index](https://pypi.python.org/pypi/pyqlearning/)

### Dependencies

- numpy: v1.13.3 or higher.
- pandas: v0.22.0 or higher.

## Demonstration: Jupyter notebook

<p align="center"><a href="search_maze_by_q_learning.ipynb" target="_top"><img src="https://github.com/chimera0/accel-brain-code/raw/pyqlearning_pandas_numpy/Reinforcement-Learning/img/maze_q_learning_result.png" width="80%" /></a></p>

I have details of this library on my Jupyter notebook: [search_maze_by_q_learning.ipynb](search_maze_by_q_learning.ipynb). This notebook demonstrates a simple maze solving algorithm based on Q-Learning. As shown in the above image, the state-action value function and parameters setting can be designed to correspond with the optimality of route.

## Demonstration: Q-Learning

[demo_maze_greedy_q_learning.py](demo_maze_greedy_q_learning.py) is a simple maze solving algorithm. `MazeGreedyQLearning` in  [devsample/maze_greedy_q_learning.py](devsample/maze_greedy_q_learning.py) is a `Concrete Class` in `Template Method Pattern` to run the Q-Learning algorithm for this task. `GreedyQLearning` in [pyqlearning/qlearning/greedy_q_learning.py](pyqlearning/qlearning/greedy_q_learning.py) is also `Concreat Class` for the epsilon-greedy-method. The `Abstract Class` that defines the skeleton of Q-Learning algorithm in the operation and declares algorithm placeholders is [pyqlearning/q_learning.py](pyqlearning/q_learning.py).  So [demo_maze_greedy_q_learning.py](demo_maze_greedy_q_learning.py) is a kind of `Client` in `Template Method Pattern`. 

This algorithm allow the *agent* to search the goal in maze by *reward value* in each point in map. 

The following is an example of map.

```
[['#' '#' '#' '#' '#' '#' '#' '#' '#' '#']
 ['#' 'S'  4   8   8   4   9   6   0  '#']
 ['#'  2  26   2   5   9   0   6   6  '#']
 ['#'  2  '@' 38   5   8   8   1   2  '#']
 ['#'  3   6   0  49   8   3   4   9  '#']
 ['#'  9   7   4   6  55   7   0   3  '#']
 ['#'  1   8   4   8   2  69   8   2  '#']
 ['#'  1   0   2   1   7   0  76   2  '#']
 ['#'  2   8   0   1   4   7   5  'G' '#']
 ['#' '#' '#' '#' '#' '#' '#' '#' '#' '#']]
```

- `#` is wall in maze.
- `S` is a start point.
- `G` is a goal.
- `@` is the agent.

In relation to reinforcement learning, the *state* of *agent* is 2-d position coordinates and the *action* is to dicide the direction of movement. Within the wall, the *agent* is movable in a cross direction and can advance by one point at a time. After moving into a new position, the *agent* can obtain a *reward*. On greedy searching, this extrinsically motivated *agent* performs in order to obtain some *reward* as high as possible. Each *reward value* is plot in map.

To see how *agent* can search and rearch the goal, run the batch program: [demo_maze_greedy_q_learning.py](demo_maze_greedy_q_learning.py)

```bash
python demo_maze_greedy_q_learning.py
```

## Demonstration: Q-Learning, loosely coupled with Deep Boltzmann Machine.

[demo_maze_deep_boltzmann_q_learning.py](demo_maze_deep_boltzmann_q_learning.py) is a demonstration of how the *Q-Learning* can be to *deepen*. A so-called *Deep Q-Network* (DQN) is meant only as an example. In this demonstration, let me cite the *Q-Learning* , loosely coupled with **Deep Boltzmann Machine** (DBM). As API Documentation of [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern) library has pointed out, DBM is functionally equivalent to stacked auto-encoder. The main function I observe is the same as dimensions reduction(or pre-training). Then the function this DBM is dimensionality reduction of *reward value* matrix.

Q-Learning, loosely coupled with Deep Boltzmann Machine (DBM), is a more effective way to solve maze. The pre-training by DBM allow Q-Learning *agent* to abstract feature of `reward value` matrix and to observe the map in a bird's-eye view. Then *agent* can reache the goal with a smaller number of trials.

To realize the power of DBM, I performed a simple experiment.

### Feature engineering

For instance, the following is a tuple of so-called *observed data points* in DBM learning.

```python
Tuple(
    'real reward value',
    'expected reward value if moving on position: left top',
    'expected reward value if moving on position: center top',
    'expected reward value if moving on position: right top',
    'expected reward value if moving on position: left middle',
    'expected reward value if moving on position: right middle',
    'expected reward value if moving on position: left bottom',
    'expected reward value if moving on position: center bottom',
    'expected reward value if moving on position: right bottom'
)
```

Then, the following is a tuple of so-called *feature points* in DBM learning.

```python
Tuple(
    'extracted feature point which can correspond the real reward value',
    'extracted feature point which can correspond expected reward value if moving on position: left top',
    'extracted feature point which can correspond expected reward value if moving on position: center top',
    'extracted feature point which can correspond expected reward value if moving on position: right top',
    'extracted feature point which can correspond expected reward value if moving on position: left middle',
    'extracted feature point which can correspond expected reward value if moving on position: right middle',
    'extracted feature point which can correspond expected reward value if moving on position: left bottom',
    'extracted feature point which can correspond expected reward value if moving on position: center bottom',
    'extracted feature point which can correspond expected reward value if moving on position: right bottom'
)
```

After pre-training, the DBM has extracted *feature points* below.

```
[['#' '#' '#' '#' '#' '#' '#' '#' '#' '#']
 ['#' 'S' 0.22186305563593528 0.22170599483791015 0.2216928599218454
  0.22164807496640074 0.22170371283788584 0.22164021608623224
  0.2218165339471332 '#']
 ['#' 0.22174745260072407 0.221880094307873 0.22174244728061343
  0.2214709292493749 0.22174626768015263 0.2216756589222596
  0.22181057818975275 0.22174525714311788 '#']
 ['#' 0.22177496678085065 0.2219122743656551 0.22187543599733664
  0.22170745588799798 0.2215226084843615 0.22153827385193636
  0.22168466277729898 0.22179391402965035 '#']
 ['#' 0.2215341770250964 0.22174315536140118 0.22143149966676515
  0.22181685688674144 0.22178215385805333 0.2212249704384472
  0.22149210148879617 0.22185413678274837 '#']
 ['#' 0.22162363223483128 0.22171313373253035 0.2217109987501002
  0.22152432841656014 0.22175562457887335 0.22176040052504634
  0.22137688854285298 0.22175365642579478 '#']
 ['#' 0.22149515807715153 0.22169199881701832 0.22169558478042856
  0.2216904005450013 0.22145368271014734 0.2217144069625017
  0.2214896100292738 0.221398594191006 '#']
 ['#' 0.22139837944992058 0.22130176116356184 0.2215414328019404
  0.22146667964656613 0.22164354506366127 0.22148685616333666
  0.22162822887193126 0.22140174437162474 '#']
 ['#' 0.22140060918518528 0.22155145714201702 0.22162929776464463
  0.22147466752374162 0.22150300682310872 0.22162775291471243
  0.2214233075299188 'G' '#']
 ['#' '#' '#' '#' '#' '#' '#' '#' '#' '#']]
```

To see how *agent* can search and rearch the goal, install [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern) library and run the batch program: [demo_maze_deep_boltzmann_q_learning.py](demo_maze_deep_boltzmann_q_learning.py)

```bash
python demo_maze_deep_boltzmann_q_learning.py
```

### Case 1: for more greedy searches

#### Map setting.
- map size: `20` * `20`.
- Start Point: (1, 1)
- End Point: (18, 18)

#### Reward value

```python
import numpy as np

map_d = 20
map_arr = np.random.rand(map_d, map_d)
map_arr += np.diag(list(range(map_d)))
```

#### Hyperparameters

- Alpha: `0.9`
- Gamma: `0.9`
- Greedy rate(epsilon): `0.75`
    * More Greedy.

#### Searching plan

- number of trials: `1000`
- Maximum Number of searches: `10000`

#### Metrics (Number of searches)

Tests show that the number of searches on the *Q-Learning* with pre-training is smaller than not with pre-training.

<table>
<thead>
<tr>
<th align="left">Number of searches</th>
<th align="left">not pre-training</th>
<th align="left">pre-training</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left">Max</td>
<td align="left">8155</td>
<td align="left">4373</td>
</tr>
<tr>
<td align="left">mean</td>
<td align="left">3753.80</td>
<td align="left">1826.0</td>
</tr>
<tr>
<td align="left">median</td>
<td align="left">3142.0</td>
<td align="left">1192.0</td>
</tr>
<tr>
<td align="left">min</td>
<td align="left">1791</td>
<td align="left">229</td>
</tr>
<tr>
<td align="left">var</td>
<td align="left">3262099.36</td>
<td align="left">2342445.78</td>
</tr>
<tr>
<td align="left">std</td>
<td align="left">1806.13</td>
<td align="left">1530.56</td>
</tr></tbody></table>

### Case 2: for less greedy searches

#### Map setting
- map size: `20` * `20`.
- Start Point: (1, 1)
- End Point: (18, 18)

#### Reward value

```python
import numpy as np

map_d = 20
map_arr = np.random.rand(map_d, map_d)
map_arr += np.diag(list(range(map_d)))
```

#### Hyperparameters

- Alpha: `0.9`
- Gamma: `0.9`
- Greedy rate(epsilon): `0.25`
    * Less Greedy.

#### Searching plan

- number of trials: `1000`
- Maximum Number of searches: `10000`

#### Metrics (Number of searches)

<table>
<thead>
<tr>
<th align="left">Number of searches</th>
<th align="left">not pre-training</th>
<th align="left">pre-training</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left">Max</td>
<td align="left">10000</td>
<td align="left">10000</td>
</tr>
<tr>
<td align="left">mean</td>
<td align="left">7136.0</td>
<td align="left">3296.89</td>
</tr>
<tr>
<td align="left">median</td>
<td align="left">9305.0</td>
<td align="left">1765.0</td>
</tr>
<tr>
<td align="left">min</td>
<td align="left">2401</td>
<td align="left">195</td>
</tr>
<tr>
<td align="left">var</td>
<td align="left">9734021.11</td>
<td align="left">10270136.10</td>
</tr>
<tr>
<td align="left">std</td>
<td align="left">3119.94</td>
<td align="left">3204.71</td>
</tr></tbody></table>

Under the assumption that the less number of searches the better, *Q-Learning*, loosely coupled with *Deep Boltzmann Machine*, is a more effective way to solve maze in not greedy mode as well as greedy mode.

### More detail demos

- [Webクローラ型人工知能：キメラ・ネットワークの仕様](https://media.accel-brain.com/_chimera-network-is-web-crawling-ai/)
    - 20001 bots are running as 20001 web-crawlers and 20001 web-scrapers.

### Related PoC

- [Webクローラ型人工知能によるパラドックス探索暴露機能の社会進化論](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/) (Japanese)
    - [プロトタイプの開発：人工知能エージェント「キメラ・ネットワーク」](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/5/#i-8)
- [深層強化学習のベイズ主義的な情報探索に駆動された自然言語処理の意味論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/) (Japanese)
    - [プロトタイプの開発：深層学習と強化学習による「排除された第三項」の推論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/4/#i-5)
- [ハッカー倫理に準拠した人工知能のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/) (Japanese)
    - [プロトタイプの開発：深層強化学習のアーキテクチャ設計](https://accel-brain.com/architectural-design-of-artificial-intelligence-conforming-to-hacker-ethics/5/#i-2)    
- [ヴァーチャルリアリティにおける動物的「身体」の物神崇拝的なユースケース](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/) (Japanese)
    - [プロトタイプの開発：「人工天使ヒューズ＝ヒストリア」](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/4/#i-6)

## Version

- 1.0.3

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0
