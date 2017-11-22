# Reinforcement Learning

These Python Scripts create a *template method pattern* for implementing a Q-learning.

## Description

Considering many variable parts and functional extensions in the Q-learning paradigm, I implemented these Python Scripts for  demonstrations of *commonality/variability analysis* in order to design the models.

### Demonstration: Q-Learning

[demo_maze_greedy_q_learning.py](demo_maze_greedy_q_learning.py) is a simple maze solving algorithm. This algorithm allow the *agent* to search the goal in maze by *reward value* in each point in map. 

The following is an example of map.

```
[['#' '#' '#' '#' '#' '#' '#' '#' '#' '#']
 ['#' 'S'  4   8   8   4   9   6   0  '#']
 ['#'  2  26   2   5   9   0   6   6  '#']
 ['#'  2  '@'  38   5   8   8   1   2  '#']
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

### Demonstration: Q-Learning, loosely coupled with Deep Boltzmann Machine.

[demo_maze_deep_boltzmann_q_learning.py](demo_maze_deep_boltzmann_q_learning.py) is a demonstration of how the *Q-Learning* can be to *deepen*. A so-called *Deep Q-Network* (DQN) is meant only as an example. In this demonstration, let me cite the *Q-Learning* , loosely coupled with **Deep Boltzmann Machine** (DBM). As API Documentation of [pydbm](https://github.com/chimera0/accel-brain-code/tree/master/Deep-Learning-by-means-of-Design-Pattern) library has pointed out, DBM is functionally equivalent to stacked auto-encoder. The main function I observe is the same as dimensions reduction(or pre-training). Then the function this DBM is dimensionality reduction of *reward value* matrix.

Q-Learning, loosely coupled with Deep Boltzmann Machine (DBM), is a more effective way to solve maze. The pre-training by DBM allow Q-Learning *agent* to abstract feature of `reward value` matrix and to observe the map in a bird's-eye view. Then *agent* can reache the goal with a smaller number of trials.

To realize the power of DBM, I performed a simple experiment.

#### Feature engineering

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

| Number of searches | not pre-training | pre-training |
|:-----------|:-----------|:-----------|
| Max       | 8155        | 4373         |
| mean     | 3753.80      | 1826.0       |
| median       | 3142.0        | 1192.0         |
| min         | 1791          | 229           |
| var       | 3262099.36       | 2342445.78       |
| std    | 1806.13     | 1530.56      |

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

| Number of searches | not pre-training | pre-training |
|:-----------|:-----------|:-----------|
| Max       | 10000        | 10000         |
| mean     | 7136.0      | 3296.89       |
| median       | 9305.0        | 1765.0         |
| min         | 2401          | 195           |
| var       | 9734021.11       | 10270136.10       |
| std    | 3119.94     | 3204.71      |

Under the assumption that the less number of searches the better, *Q-Learning*, loosely coupled with *Deep Boltzmann Machine*, is a more effective way to solve maze in not greedy mode as well as greedy mode.

### More detail demos

- [Webクローラ型人工知能：キメラ・ネットワークの仕様](https://media.accel-brain.com/_chimera-network-is-web-crawling-ai/)
    - 20001 bots are running as 20001 web-crawlers and 20001 web-scrapers.

### Related Case Studies

- [Webクローラ型人工知能によるパラドックス探索暴露機能の社会進化論](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/)
    - [ケーススタディ：人工知能エージェント「キメラ・ネットワーク」](https://accel-brain.com/social-evolution-of-exploration-and-exposure-of-paradox-by-web-crawling-type-artificial-intelligence/5/#i-8)
- [深層強化学習のベイズ主義的な情報探索に駆動された自然言語処理の意味論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/)
    - [ケーススタディ：深層学習と強化学習による「排除された第三項」の推論](https://accel-brain.com/semantics-of-natural-language-processing-driven-by-bayesian-information-search-by-deep-reinforcement-learning/4/#i-5)
- [ヴァーチャルリアリティにおける動物的「身体」の物神崇拝的なユースケース](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/)
    - [ケーススタディ：「人工天使ヒューズ＝ヒストリア」](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/4/#i-6)

## Version
- 1.0

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0

## Requires

- Python3.4.4
