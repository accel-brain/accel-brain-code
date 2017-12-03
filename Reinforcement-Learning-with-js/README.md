# Reinforcement Learning with JavaScript

These JavaScript modules are library to implement Reinforcement Learning, especially for Q-Learning.

## Description

These modules are functionally equivalent to Python Scripts in [pyqlearning](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning). Considering many variable parts and functional extensions in the Q-learning paradigm, I implemented these scripts for demonstrations of *commonality/variability analysis* in order to design the models.

## Installation

### Source code

The source code is currently hosted on GitHub.

- [accel-brain-code/Reinforcement-Learning-with-js](https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning-with-js)

## Demonstration: Autocompletion

- [demo_autocompletion.html](demo_autocompletion.html)

### Code sample

The function of autocompletion is a kind of natural language processing. Load follow JavaScript files in [devsample](devsample/). These scripts are functionally equivalent to Python Scripts in [pysummarization](https://github.com/chimera0/accel-brain-code/tree/master/Automatic-Summarization).

```
<script type="text/javascript" src="devsample/nlpbase.js"></script>
<script type="text/javascript" src="devsample/ngram.js"></script>
```

The modules of autocompletion depend on [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) (v0.2). Load this JavaScript file.

```
<script type="text/javascript" src="dependencies/tiny_segmenter-0.2.js"></script>
```

And `Q-Learning` modules are to be included.

```html
<script type="text/javascript" src="jsqlearning/qlearning.js"></script>
<script type="text/javascript" src="jsqlearning/qlearning/boltzmann.js"></script>
<script type="text/javascript" src="jsqlearning/qlearning/boltzmann/autocompletion.js"></script>
```

Initialize NLP modules.

```js
// The number of n-gram.
var n = 2;
// The function of n-gram.
var n_gram = new Ngram();
// Base class of NLP for tokenization.
var nlp_base = new NlpBase();

// The function of autocompletion algorithm.
var autocompletion = new Autocompletion(
    nlp_base,
    n_gram,
    n
);
```

And, setup hyperparameters in Q-Learning and initialize.

```js
// Time rate in boltzmann distribution.
taime_rate = 0.001;

// Alpha value in Q-Learning algorithm.
alpha_value = 0.5;
// Gamma value in Q-Learning algorithm.
gamma_value = 0.5;
// The number of learning.
limit = 10000;

// The algorithm of boltzmann distribution.
var boltzmann = new Boltzmann(
    autocompletion,
    {
        "time_rate": time_rate
    }
);

// Base class of Q-Learning.
var q_learning = new QLearning(
    boltzmann,
    {
        "alpha_value": alpha_value,
        "gamma_value": gamma_value
    }
);

```

Set learned data.

```js
// Learned data.
first_learned_data = "hogehogehogefugafuga";

// Pre training for first user's typing.
autocompletion_.pre_training(
    q_learning,
    first_learned_data
);
```

Execute recursive learning in loop control structure or recursive call.

```js
// User's typing.
input_document = "hogefuga";

// Extract state in input_document.
var state_key = autocompletion.lap_extract_ngram(
    q_learning,
    input_document
);

// Learning.
q_learning.learn(state_key, limit);

// Predict next token.
var next_action_list = q_learning.extract_possible_actions(
    state_key
);
var action_key = q_learning.select_action(
    state_key,
    next_action_list
);

// Compute reward value.
var reward_value = q_learning.observe_reward_value(
    state_key,
    action_key
);

// Compute Q-Value.
var q_value = q_learning.extract_q_dict(
    state_key,
    action_key
);

// Pre training for next user's typing.
autocompletion_.pre_training(
    q_learning,
    input_document
);
```

## More detail demos

- @TODO(chimera0)

## Related PoC

- @TODO(chimera0)

## Version

- 1.0.1

## Author

- chimera0(RUM)

## Author URI

- [http://accel-brain.com/](http://accel-brain.com/)

## License

- GNU General Public License v2.0
